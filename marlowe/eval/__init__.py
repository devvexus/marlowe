"""Evaluation harness.

Two families of metric, and the project needs both:

* :mod:`marlowe.eval.kl` -- teacher-forced fidelity against the bf16 parent. Cheap, stable,
  comparable across checkpoints.
* :mod:`marlowe.eval.repetition` -- generation-based loop detection. The only thing that can
  see the artefact this project exists to fix.

KL is measured teacher-forced on fixed text. Circling is an autoregressive failure that
appears only when the model samples its own continuations, so a checkpoint can have
excellent KL and still loop. Shipping on KL alone ships a looping model.
"""

from __future__ import annotations
