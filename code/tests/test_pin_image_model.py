"""The generator is pinned by the harness, not requested by the model.

Omitting `model` on generate_image does not use the model the policy names — it
uses the server's default, and the call succeeds. A benchmark arm can therefore
run to completion on the wrong generator with nothing reporting a substitution.
Measured on real runs at 2.4% of items for one reasoner and ~10% for another,
which is why this cannot be left to instruction-following.
"""

import pytest

from continuum.harnesses.agent_loop import ImageRegistry, prepare_call_args

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def registry():
    return ImageRegistry()


def test_pin_supplies_a_model_the_agent_omitted(registry) -> None:
    call = prepare_call_args({"prompt": "a cat"}, registry,
                             pin_model="flux2-klein", tool_name="generate_image")
    assert call["model"] == "flux2-klein"


def test_pin_overrides_a_model_the_agent_chose(registry) -> None:
    # Not only the omitted case: an arm measuring one model must not be handed
    # another because the agent preferred it.
    call = prepare_call_args({"prompt": "a cat", "model": "qwen-image"}, registry,
                             pin_model="flux2-klein", tool_name="generate_image")
    assert call["model"] == "flux2-klein"


def test_editing_is_pinned_too(registry) -> None:
    # The default editor is a different model from the default generator, so an
    # unpinned edit silently changes arms mid-run.
    img = registry.register(PNG)
    call = prepare_call_args({"image": img, "prompt": "make it night"}, registry,
                             pin_model="flux2-klein", tool_name="edit_image")
    assert call["model"] == "flux2-klein"


def test_not_forced_onto_tools_without_a_model_parameter(registry) -> None:
    # inspect_region's schema has no model; sending one is a validation error.
    img = registry.register(PNG)
    call = prepare_call_args({"image": img, "roi": [0, 0, 10, 10]}, registry,
                             pin_model="flux2-klein", tool_name="inspect_region")
    assert "model" not in call


def test_unpinned_leaves_the_choice_alone(registry) -> None:
    assert "model" not in prepare_call_args({"prompt": "a cat"}, registry)
    kept = prepare_call_args({"prompt": "a cat", "model": "qwen-image"}, registry)
    assert kept["model"] == "qwen-image"
