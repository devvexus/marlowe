"""Synthetic mini-checkpoint fixtures.

Realistic tensor naming, tiny tensors. The point is to exercise the naming, renumbering and
config-rewrite logic end to end without 56 GB of weights.

The checkpoint deliberately contains **three** ``.layers.`` namespaces:

* ``model.language_model.layers.``  -- the text decoder (the one that must be selected)
* ``model.visual.layers.``          -- the vision tower, a different depth (must be ignored)
* ``model.mtp.layers.``             -- the MTP draft head (must be ignored)

Prefix detection that took the first match, or that keyed on anything but the layer count,
would grab the wrong stack. That is the bug this fixture exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

TEXT_PREFIX = "model.language_model.layers."
VISION_PREFIX = "model.visual.layers."
MTP_PREFIX = "model.mtp.layers."

N_TEXT_LAYERS = 12  # 3 periods of 4
N_VISION_LAYERS = 27
HIDDEN = 8


def linear_layer_tensors(i: int, prefix: str = TEXT_PREFIX) -> list[str]:
    return [
        f"{prefix}{i}.linear_attn.q_proj.weight",
        f"{prefix}{i}.linear_attn.k_proj.weight",
        f"{prefix}{i}.linear_attn.v_proj.weight",
        f"{prefix}{i}.linear_attn.g_proj.weight",
        f"{prefix}{i}.linear_attn.out_proj.weight",
        f"{prefix}{i}.linear_attn.conv1d.weight",
        f"{prefix}{i}.linear_attn.conv1d.bias",
        f"{prefix}{i}.linear_attn.norm.weight",
        f"{prefix}{i}.linear_attn.b_proj.weight",
        f"{prefix}{i}.linear_attn.a_proj.weight",
    ]


def full_layer_tensors(i: int, prefix: str = TEXT_PREFIX) -> list[str]:
    return [
        f"{prefix}{i}.self_attn.q_proj.weight",
        f"{prefix}{i}.self_attn.k_proj.weight",
        f"{prefix}{i}.self_attn.v_proj.weight",
        f"{prefix}{i}.self_attn.o_proj.weight",
        f"{prefix}{i}.self_attn.q_norm.weight",
        f"{prefix}{i}.self_attn.k_norm.weight",
    ]


def shared_layer_tensors(i: int, prefix: str = TEXT_PREFIX) -> list[str]:
    return [
        f"{prefix}{i}.mlp.gate_proj.weight",
        f"{prefix}{i}.mlp.up_proj.weight",
        f"{prefix}{i}.mlp.down_proj.weight",
        f"{prefix}{i}.input_layernorm.weight",
        f"{prefix}{i}.post_attention_layernorm.weight",
    ]


def layer_types_for(n_layers: int, attention_type: str = "full_attention") -> list[str]:
    """n_layers as repeated [linear, linear, linear, attention]."""
    out: list[str] = []
    for i in range(n_layers):
        out.append("linear_attention" if (i + 1) % 4 else attention_type)
    return out


def tensor_names(layer_types: list[str], prefix: str = TEXT_PREFIX) -> list[str]:
    names: list[str] = []
    for i, t in enumerate(layer_types):
        names += shared_layer_tensors(i, prefix)
        names += (
            linear_layer_tensors(i, prefix)
            if t == "linear_attention"
            else full_layer_tensors(i, prefix)
        )
    return names


def build_config(layer_types: list[str]) -> dict[str, Any]:
    """A config shaped like the real one: nested text_config, vision_config, MTP, interval."""
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": HIDDEN,
            "intermediate_size": 4 * HIDDEN,
            "vocab_size": 64,
            "tie_word_embeddings": False,
            "num_hidden_layers": len(layer_types),
            "layer_types": list(layer_types),
            # The field that must not survive surgery: some runtimes regenerate layer_types
            # from it instead of reading the explicit list.
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attn_output_gate": True,
        },
        "vision_config": {"depth": N_VISION_LAYERS, "hidden_size": 6, "out_hidden_size": HIDDEN},
        "deepstack_visual_indexes": [],
    }


def write_checkpoint(path: Path, layer_types: list[str], *, sharded: bool = True) -> Path:
    """Write a real safetensors checkpoint with all three .layers. namespaces."""
    import torch
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    names = tensor_names(layer_types)
    names += [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    # decoys
    for i in range(N_VISION_LAYERS):
        names += [
            f"{VISION_PREFIX}{i}.attn.qkv.weight",
            f"{VISION_PREFIX}{i}.mlp.fc1.weight",
        ]
    names += [
        f"{MTP_PREFIX}0.self_attn.q_proj.weight",
        f"{MTP_PREFIX}0.mlp.gate_proj.weight",
        "model.mtp.norm.weight",
    ]

    tensors = {n: torch.ones(2, 2, dtype=torch.float32) * (h % 7 + 1)
               for h, n in enumerate(names)}

    if sharded:
        keys = list(tensors)
        half = len(keys) // 2
        groups = {"model-00001-of-00002.safetensors": keys[:half],
                  "model-00002-of-00002.safetensors": keys[half:]}
        weight_map: dict[str, str] = {}
        for fn, ks in groups.items():
            save_file({k: tensors[k] for k in ks}, str(path / fn), metadata={"format": "pt"})
            for k in ks:
                weight_map[k] = fn
        with (path / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": 4 * 4 * len(tensors)},
                       "weight_map": weight_map}, f, indent=2)
    else:
        save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})

    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(build_config(layer_types), f, indent=2)
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "generation_config.json").write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def mini_layer_types() -> list[str]:
    return layer_types_for(N_TEXT_LAYERS)


@pytest.fixture
def mini_checkpoint(tmp_path: Path, mini_layer_types: list[str]) -> Path:
    return write_checkpoint(tmp_path / "mini-27b", mini_layer_types)


@pytest.fixture
def mini_names(mini_layer_types: list[str]) -> list[str]:
    return tensor_names(mini_layer_types)
