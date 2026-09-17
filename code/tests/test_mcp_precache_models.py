import continuum.models
from continuum import mcp_server
from continuum.models import (
    FIRERED_REPO,
    FLUX2_KLEIN_REPO,
    QWEN_IMAGE_EDIT_REPO,
    QWEN_IMAGE_REPO,
)


def test_precache_includes_all_exposed_model_repos(monkeypatch) -> None:
    repos: list[str] = []

    monkeypatch.setattr(continuum.models, "ensure_local", lambda repo, token=None: repos.append(repo))

    mcp_server._precache_models()

    assert FLUX2_KLEIN_REPO in repos
    assert QWEN_IMAGE_REPO in repos
    assert FIRERED_REPO in repos
    assert QWEN_IMAGE_EDIT_REPO in repos
