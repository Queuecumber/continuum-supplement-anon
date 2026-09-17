"""An id the model invented should cost a turn, not the run.

Weaker reasoners name an image before producing one -- `edit_image(image="img_0")`
as the very first call. That used to raise KeyError out of the loop and fail the
item at step 1, which is a harness fault dressed up as a model failure: observed
across every cell of the benchmark matrix, worst on the smallest models.
"""

import pytest

from continuum.harnesses.agent_loop import ImageRegistry, UnknownImageId, prepare_call_args

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def test_unknown_id_is_recoverable_not_fatal() -> None:
    registry = ImageRegistry()
    registry.register(PNG)
    with pytest.raises(UnknownImageId) as caught:
        prepare_call_args({"image": "img_9"}, registry, tool_name="edit_image")
    assert not isinstance(caught.value, KeyError)


def test_message_names_what_does_exist() -> None:
    registry = ImageRegistry()
    first = registry.register(PNG)
    with pytest.raises(UnknownImageId) as caught:
        prepare_call_args({"image": "img_9"}, registry, tool_name="edit_image")
    # Listing the ids is the difference between a correctable turn and a guess.
    assert first in caught.value.message()


def test_empty_registry_says_so() -> None:
    # The real case: an id named before anything has been generated.
    with pytest.raises(UnknownImageId) as caught:
        prepare_call_args({"image": "img_0"}, ImageRegistry(), tool_name="edit_image")
    assert "none yet" in caught.value.message()


def test_valid_ids_are_unaffected() -> None:
    registry = ImageRegistry()
    img = registry.register(PNG)
    call = prepare_call_args({"image": img, "prompt": "x"}, registry, tool_name="edit_image")
    assert call["image"]["type"] == "image"
