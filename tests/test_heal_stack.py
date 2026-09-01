"""Live compatibility check for the Stage 6 training stack.

Not an import test. Builds a tiny but architecturally real ``qwen3_5`` model -- hybrid
layer_types, Gated DeltaNet mixers, gated GQA with the fused output gate -- loads it NF4
through the same :func:`marlowe.heal.load_student` the pipeline uses, wraps it in LoRA, and
runs a real forward/backward/step against the real top-K KL loss.

The point is that peft reaches into transformers internals, and this project just jumped
transformers 5.1 -> 5.16 with peft and bitsandbytes installed against the old version.
Discovering that at Stage 6 hour 3 costs three days.

Marked ``slow``: needs a CUDA device and bitsandbytes. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


requires_gpu = pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA device")


TINY_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


def tiny_qwen35_config(n_layers: int = 8, mtp: int = 0) -> dict[str, Any]:
    """A real qwen3_5_text config at toy dimensions.

    Head dims are kept at the published values where the architecture constrains them
    (``linear_key_head_dim``/``linear_value_head_dim``) so the DeltaNet path is exercised
    rather than degenerate.
    """
    types = (TINY_LAYER_TYPES * ((n_layers // 4) + 1))[:n_layers]
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": 256,
        "intermediate_size": 512,
        "vocab_size": 1024,
        "tie_word_embeddings": False,
        "num_hidden_layers": n_layers,
        "layer_types": types,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attn_output_gate": True,
        "linear_num_value_heads": 8,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": mtp,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "partial_rotary_factor": 0.25,
        "torch_dtype": "bfloat16",
    }


def build_tiny_model(path: Path, *, n_layers: int = 8, mtp: int = 0) -> Path:
    """Materialise a randomly-initialised tiny qwen3_5 checkpoint on disk."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(tiny_qwen35_config(n_layers, mtp), f, indent=2)

    cfg = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "qwen35-tiny")


class TestArchitectureIsSupported:
    def test_transformers_knows_qwen3_5(self) -> None:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        assert "qwen3_5" in CONFIG_MAPPING_NAMES, (
            "transformers cannot parse this architecture; Stages 3, 5 and 6 all load the "
            "model through AutoModelForCausalLM and would fail immediately"
        )
        assert "qwen3_5_text" in CONFIG_MAPPING_NAMES

    def test_pinned_minimum_is_installed(self) -> None:
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) >= Version("5.16"), (
            "pyproject pins transformers>=5.16 because 5.1 cannot parse qwen3_5"
        )

    def test_tiny_model_builds_with_the_real_layout(self, tiny_model: Path) -> None:
        from marlowe.arch import Layout, load_config

        layout = Layout.from_config(load_config(tiny_model))
        assert layout.n_layers == 8
        assert layout.n_attention == 2
        assert layout.n_removable_type == 6


class TestZeroMtpLoads:
    """``mtp_num_hidden_layers: 0`` must round-trip through from_pretrained.

    The GGUF converter needed an explicit ``--no-mtp`` because it treats 0 as "unspecified,
    discover it from the tensors". Stage 6 loads the pruned student through
    ``from_pretrained``, so the same zero-versus-unspecified ambiguity has to be ruled out on
    that path too.
    """

    def test_config_roundtrips_with_zero(self, tmp_path: Path) -> None:
        from transformers import AutoConfig

        from marlowe.arch import text_config

        p = build_tiny_model(tmp_path / "zero", mtp=0)
        cfg = AutoConfig.from_pretrained(p)
        t = text_config(cfg)
        assert getattr(t, "mtp_num_hidden_layers", 0) == 0

    @requires_gpu
    def test_model_loads_and_runs_with_zero_mtp(self, tmp_path: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        p = build_tiny_model(tmp_path / "zero-run", mtp=0)
        model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
        with torch.no_grad():
            out = model(input_ids=torch.randint(0, 1024, (1, 64), device="cuda"))
        assert out.logits.shape == (1, 64, 1024)
        assert torch.isfinite(out.logits).all(), "zero-MTP config produced non-finite logits"

    def test_surgery_output_declares_zero(self, mini_checkpoint, tmp_path: Path) -> None:
        """What --drop-mtp actually writes, checked against the real surgery path."""
        from marlowe.arch import load_config, text_config
        from marlowe.surgery import run_surgery

        out = tmp_path / "child"
        run_surgery(mini_checkpoint, out, [5], drop_mtp=True, drop_vision=True)
        t = text_config(load_config(out))
        assert t["mtp_num_hidden_layers"] == 0
        assert "mtp_use_dedicated_embeddings" not in t


@requires_gpu
class TestQLoRAStack:
    """peft + bitsandbytes + transformers 5.16, exercised end to end."""

    def test_bitsandbytes_imports_and_reports_a_version(self) -> None:
        import bitsandbytes

        assert bitsandbytes.__version__

    def test_load_student_returns_a_trainable_4bit_model(self, tiny_model: Path) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import load_student

        cfg = HealConfig(lora_rank=8, lora_alpha=16)
        model = load_student(str(tiny_model), cfg)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "no trainable parameters; LoRA did not attach"
        assert all("lora" in n.lower() for n in trainable), (
            f"non-LoRA parameters are trainable: {[n for n in trainable if 'lora' not in n][:5]}"
        )

    def test_four_bit_quantisation_actually_applied(self, tiny_model: Path) -> None:
        """The residency assertion is only meaningful if bnb really quantised the linears."""
        import bitsandbytes as bnb

        from marlowe.config import HealConfig
        from marlowe.heal import load_student

        model = load_student(str(tiny_model), HealConfig(lora_rank=8))
        n4bit = sum(
            1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)
        )
        assert n4bit > 0, "no Linear4bit modules; the quantization config was ignored"

    def test_one_optimizer_step_against_the_real_loss(self, tiny_model: Path) -> None:
        """Forward, top-K KL backward, step. The thing that must not break at hour 3."""
        import torch

        from marlowe.config import HealConfig
        from marlowe.heal import chunked_kl_loss, load_student
        from marlowe.score import find_decoder

        cfg = HealConfig(lora_rank=8, lora_alpha=16)
        model = load_student(str(tiny_model), cfg)
        decoder = find_decoder(model)
        lm_head = model.get_output_embeddings()
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=1e-4)

        seq, top_k = 64, 8
        ids = torch.randint(0, 1024, (1, seq), device="cuda")
        t_idx = torch.randint(0, 1024, (1, seq, top_k), device="cuda")
        t_lp = torch.log_softmax(torch.randn(1, seq, top_k, device="cuda"), dim=-1)

        model.train()
        out = decoder(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        loss, diag = chunked_kl_loss(lm_head, hidden, t_idx, t_lp, chunk=32)

        assert torch.isfinite(loss), "loss is not finite"
        assert loss.item() >= 0, "forward KL must be non-negative"
        assert 0.0 <= diag["captured_mass"] <= 1.0001

        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        assert grads, "no gradients reached the LoRA parameters"
        assert any(g.abs().sum().item() > 0 for g in grads), "all gradients are zero"

        before = params[0].detach().clone()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        assert not torch.equal(before, params[0]), "optimizer step did not change the weights"

    def test_gradient_checkpointing_path(self, tiny_model: Path) -> None:
        """Gradient checkpointing is what makes 2048 fit; it must not silently detach."""
        import torch

        from marlowe.config import HealConfig
        from marlowe.heal import chunked_kl_loss, load_student
        from marlowe.score import find_decoder

        model = load_student(
            str(tiny_model), HealConfig(lora_rank=8, gradient_checkpointing=True)
        )
        decoder = find_decoder(model)
        model.train()
        ids = torch.randint(0, 1024, (1, 32), device="cuda")
        out = decoder(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        loss, _ = chunked_kl_loss(
            model.get_output_embeddings(),
            hidden,
            torch.randint(0, 1024, (1, 32, 4), device="cuda"),
            torch.log_softmax(torch.randn(1, 32, 4, device="cuda"), dim=-1),
            chunk=16,
        )
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert any(p.grad is not None for p in trainable), (
            "gradient checkpointing broke the graph; no LoRA gradients"
        )

    def test_adapters_save_and_reload(self, tiny_model: Path, tmp_path: Path) -> None:
        """Resumability depends on this: a 3-day run must survive a crash."""
        from marlowe.config import HealConfig
        from marlowe.heal import load_student

        model = load_student(str(tiny_model), HealConfig(lora_rank=8))
        ckpt = tmp_path / "checkpoint-1000"
        model.save_pretrained(str(ckpt))
        assert (ckpt / "adapter_config.json").exists()
        assert any(ckpt.glob("adapter_model.*"))

        model.load_adapter(str(ckpt), adapter_name="reloaded", is_trainable=True)
        assert "reloaded" in model.peft_config
