import pytest

from continuum.backends.lora import apply_loras, normalize_lora_specs, parse_lora_source

# ---- source parsing --------------------------------------------------------


def test_parse_repo_id_returns_repo_and_no_weight() -> None:
    repo, weight = parse_lora_source("someuser/cool-lora")
    assert repo == "someuser/cool-lora"
    assert weight is None


def test_parse_repo_with_explicit_file() -> None:
    repo, weight = parse_lora_source("someuser/cool-lora/adapter.safetensors")
    assert repo == "someuser/cool-lora"
    assert weight == "adapter.safetensors"


def test_parse_hf_blob_url_strips_kind_and_ref() -> None:
    repo, weight = parse_lora_source(
        "https://huggingface.co/someuser/cool-lora/blob/main/sub/adapter.safetensors"
    )
    assert repo == "someuser/cool-lora"
    assert weight == "sub/adapter.safetensors"


def test_parse_bare_url_repo_only() -> None:
    repo, weight = parse_lora_source("https://huggingface.co/someuser/cool-lora")
    assert repo == "someuser/cool-lora"
    assert weight is None


def test_parse_local_file(tmp_path) -> None:
    f = tmp_path / "my.safetensors"
    f.write_bytes(b"\x00")
    repo, weight = parse_lora_source(str(f))
    assert repo == str(f)
    assert weight is None


def test_parse_rejects_bare_word() -> None:
    with pytest.raises(ValueError):
        parse_lora_source("notarepo")


# ---- spec normalization ----------------------------------------------------


def test_normalize_string_entries_default_scale() -> None:
    specs = normalize_lora_specs(["a/b", "c/d"])
    assert [s["repo"] for s in specs] == ["a/b", "c/d"]
    assert [s["scale"] for s in specs] == [1.0, 1.0]
    assert [s["name"] for s in specs] == ["lora_0", "lora_1"]


def test_normalize_dict_entry_with_scale_and_weight() -> None:
    specs = normalize_lora_specs([{"source": "a/b", "scale": 0.7, "weight_name": "x.safetensors"}])
    assert specs[0]["repo"] == "a/b"
    assert specs[0]["scale"] == 0.7
    assert specs[0]["weight_name"] == "x.safetensors"


def test_normalize_loraspec_object_entry() -> None:
    from continuum.server.schema import LoraSpec

    specs = normalize_lora_specs([LoraSpec(source="a/b/x.safetensors", scale=0.5)])
    assert specs[0]["repo"] == "a/b"
    assert specs[0]["weight_name"] == "x.safetensors"
    assert specs[0]["scale"] == 0.5


def test_normalize_none_is_empty() -> None:
    assert normalize_lora_specs(None) == []


def test_normalize_missing_source_raises() -> None:
    with pytest.raises(ValueError):
        normalize_lora_specs([{"scale": 1.0}])


# ---- adapter state machine on a cached pipeline ----------------------------


class _DummyPipe:
    def __init__(self):
        self.loads = []
        self.unloads = 0
        self.adapters = None

    def load_lora_weights(self, repo, **kw):
        self.loads.append((repo, kw.get("weight_name"), kw.get("adapter_name")))

    def unload_lora_weights(self):
        self.unloads += 1

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights) if adapter_weights else None)


class _Owner:
    pass


def test_apply_loads_and_sets_adapter() -> None:
    owner, pipe = _Owner(), _DummyPipe()
    apply_loras(owner, pipe, normalize_lora_specs([{"source": "a/b", "scale": 0.8}]))
    assert pipe.loads == [("a/b", None, "lora_0")]
    assert pipe.adapters == (["lora_0"], [0.8])
    assert pipe.unloads == 0


def test_reapplying_same_set_is_noop() -> None:
    owner, pipe = _Owner(), _DummyPipe()
    specs = normalize_lora_specs(["a/b"])
    apply_loras(owner, pipe, specs)
    apply_loras(owner, pipe, specs)
    assert len(pipe.loads) == 1
    assert pipe.unloads == 0


def test_switching_sets_unloads_then_reloads() -> None:
    owner, pipe = _Owner(), _DummyPipe()
    apply_loras(owner, pipe, normalize_lora_specs(["a/b"]))
    apply_loras(owner, pipe, normalize_lora_specs(["c/d", "e/f"]))
    assert pipe.unloads == 1
    assert pipe.loads[-2:] == [("c/d", None, "lora_0"), ("e/f", None, "lora_1")]
    assert pipe.adapters == (["lora_0", "lora_1"], [1.0, 1.0])


def test_empty_after_nonempty_unloads() -> None:
    owner, pipe = _Owner(), _DummyPipe()
    apply_loras(owner, pipe, normalize_lora_specs(["a/b"]))
    apply_loras(owner, pipe, [])
    assert pipe.unloads == 1
    assert owner._active_lora_sig is None
