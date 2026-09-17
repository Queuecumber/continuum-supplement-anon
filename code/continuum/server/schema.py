"""Shared tool parameter types.

Both views import these so enum values, structured shapes, and defaults stay
in sync (the parity test pins that). Per-parameter PROSE is the docstring
Args' job — it is the single source the agent reads from the tool description,
so these aliases carry only the enum/shape constraint, not a Field
description. The exception is a composite type like ``LoraSpec``: its
per-field descriptions document the object's internals, which a docstring
Args line cannot reach.
"""

from typing import Literal

from pydantic import BaseModel, Field


class LoraSpec(BaseModel):
    """One LoRA adapter to load onto the pipeline."""

    source: str = Field(
        description=(
            "HF repo id (owner/name), HF URL, or local .safetensors path. A "
            "path or URL ending in a file selects that file within the repo."
        )
    )
    scale: float = Field(
        1.0,
        description=(
            "Adapter strength. 1.0 is full weight; some adapters need a "
            "specific value to take visible effect."
        ),
    )
    weight_name: str | None = Field(
        None,
        description=("Explicit weight file within a multi-file repo; usually inferred from source."),
    )


# Enum-only aliases — descriptions live in each tool's docstring Args.
GenerateImageModel = Literal["flux2-dev", "flux2-klein", "flux1-dev", "qwen-image"]

EditImageModel = Literal["firered", "flux2-dev", "flux2-klein", "qwen-image"] | None

LoraList = list[LoraSpec] | None

InspectRegionRoi = str | list[float] | dict[str, float]
