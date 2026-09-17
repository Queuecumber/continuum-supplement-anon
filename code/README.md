# Install and run

Requires Python 3.10+, a CUDA-capable GPU, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run the commands below from `code/`.

## Install

```bash
uv sync --extra agent
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

The token must have access to the model repositories you use. Weights download on first use.

## Start the server

```bash
uv run continuum-mcp --host 127.0.0.1 --port 8742
```

## Interactive use with Pi

With Node.js installed, open another terminal in `code/`:

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
npm --prefix pi-extension ci
pi install "$PWD"
CONTINUUM_MCP_URL=http://127.0.0.1:8742/mcp pi
```

In [Pi](https://github.com/earendil-works/pi/tree/main/packages/coding-agent), use `/login` to authenticate and `/model` to select a vision-capable model.

## Batch use

In another terminal in `code/`, set your agent endpoint credentials:

```bash
export OPENAI_API_KEY="YOUR_AGENT_API_KEY"
uv run continuum-batch-agent \
  --mcp-url http://127.0.0.1:8742/mcp \
  --prompt "A red bicycle leaning against a blue wall." \
  --model "YOUR_MODEL_ID" --base-url "https://YOUR_ENDPOINT/v1" \
  --agents-md benchmarks/AGENTS.md --image-model flux2-klein \
  --output-dir runs/example
```

For a prompt list, replace `--prompt "..."` with `--manifest benchmarks/unigenbench_en.jsonl`.
