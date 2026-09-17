from continuum.mcp_server import (
    _canonical_edit_model_name,
    _canonical_generate_model_name,
    _resolve_edit_defaults,
    _resolve_generate_defaults,
)
from continuum.models import (
    canonical_image_model_name,
    is_flux1_model,
    is_flux2_klein_model,
    is_flux2_model,
    is_qwen_image_edit_model,
    is_qwen_image_model,
    normalize_image_model_name,
)


def test_flux2_aliases() -> None:
    assert is_flux2_model("flux2")
    assert is_flux2_model("FLUX.2-dev")
    assert is_flux2_model("flux-2")
    assert not is_flux2_model("flux2-klein")


def test_flux1_dev_generate_aliases_and_resolution() -> None:
    for alias in ("flux1", "flux-1", "FLUX.1", "flux1-dev", "flux.1-dev"):
        assert is_flux1_model(alias)
        assert _canonical_generate_model_name(alias) == "flux1-dev"
    # Distinct from flux2 and flux1-fill.
    assert not is_flux1_model("flux2")
    assert not is_flux1_model("flux1-fill")
    assert not is_flux1_model("fill")
    # Generate-only: FLUX.1-dev is not in the edit-resolution set.
    assert _canonical_generate_model_name("flux1") == "flux1-dev"
    # FLUX.1-dev's generate defaults.
    assert _resolve_generate_defaults("flux1-dev", None, None) == (28, 3.5)


def test_flux2_klein_aliases() -> None:
    assert is_flux2_klein_model("klein")
    assert is_flux2_klein_model("flux2_klein")
    assert is_flux2_klein_model("FLUX.2-klein-4B")
    assert is_flux2_klein_model("flux2-klein-base-9b")
    assert not is_flux2_klein_model("flux2")


def test_qwen_image_edit_aliases() -> None:
    assert is_qwen_image_edit_model("qwen")
    assert is_qwen_image_edit_model("qwen-image")
    assert is_qwen_image_edit_model("qwen_image_edit")
    assert is_qwen_image_edit_model("Qwen-Image-Edit-2511")
    assert not is_qwen_image_edit_model("firered")


def test_qwen_image_generate_aliases() -> None:
    assert is_qwen_image_model("qwen-image")
    assert is_qwen_image_model("qwen_image_2512")
    assert is_qwen_image_model("qwen-image-generate")
    assert not is_qwen_image_model("qwen-image-edit")


def test_normalize_image_model_name_defaults_to_flux2() -> None:
    assert normalize_image_model_name(None) == "flux2"
    assert normalize_image_model_name(" QWEN_IMAGE_EDIT ") == "qwen-image-edit"


def test_canonical_image_model_names() -> None:
    assert canonical_image_model_name("klein") == "flux2-klein"
    assert canonical_image_model_name("qwen") == "qwen-image-edit"
    assert canonical_image_model_name("qwen-image") == "qwen-image-edit"
    assert canonical_image_model_name("fire-red") == "firered"


def test_generate_and_edit_canonicalize_qwen_image_differently() -> None:
    assert _canonical_generate_model_name("flux2-dev") == "flux2"
    assert _canonical_generate_model_name("qwen-image") == "qwen-image-generate"
    assert _canonical_edit_model_name("qwen-image") == "qwen-image-edit"


def test_edit_defaults_are_model_specific_when_omitted() -> None:
    assert _resolve_edit_defaults("flux2", None, None) == (30, 4.0)
    assert _resolve_edit_defaults("flux2-klein", None, None) == (4, 1.0)
    assert _resolve_edit_defaults("firered", None, None) == (30, 4.0)
    assert _resolve_edit_defaults("qwen-image-edit", None, None) == (40, 4.0)


def test_generate_defaults_are_model_specific_when_omitted() -> None:
    assert _resolve_generate_defaults("flux2", None, None) == (30, 4.0)
    assert _resolve_generate_defaults("klein", None, None) == (4, 1.0)
    assert _resolve_generate_defaults("qwen-image", None, None) == (50, 4.0)


def test_edit_defaults_preserve_explicit_values() -> None:
    assert _resolve_edit_defaults("qwen-image-edit", 12, 1.25) == (12, 1.25)


def test_registry_integrity() -> None:
    from continuum.models import _EDIT_RESOLUTION_ORDER, IMAGE_BACKENDS

    # Every edit-resolution key is a registered backend.
    assert set(_EDIT_RESOLUTION_ORDER) <= set(IMAGE_BACKENDS)
    for key, spec in IMAGE_BACKENDS.items():
        assert spec.key == key
        assert spec.repo
        assert callable(spec.build)
        # Aliases are stored pre-normalized.
        assert all(a == normalize_image_model_name(a) for a in spec.aliases)
        # The canonical key resolves to itself (except generate-side keys,
        # whose aliases overlap the edit registry by design).
        if key in _EDIT_RESOLUTION_ORDER:
            assert canonical_image_model_name(key) == key
