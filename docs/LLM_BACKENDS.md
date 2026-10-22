# LLM backends: MiniMax (used in the paper) and DeepSeek (optional second family)

The paper's LLM arms use MiniMax-M3.1-Flash (2.5% coverage) and MiniMax-M3
(19.3% coverage). The two are the same model family; the paper explicitly does
not claim a cross-family model effect. DeepSeek can be added as a genuinely
different family, but **the DeepSeek key is not stored in this repository** and
must be supplied through environment variables.

## How the adapter works

`src/network_evolve/llm_backend.py` selects the request format from
`NETWORK_EVOLVE_LLM_API_STYLE` or from the model/base URL:

- `anthropic` (default): `POST {base}/messages` (MiniMax/MCode);
- `openai`: `POST {base}/chat/completions` (DeepSeek/OpenAI-compatible);
- a model name containing `deepseek` or a base URL containing `deepseek`
  automatically selects `openai`.

The DeepSeek path sends a standard chat-completions body, parses
`choices[0].message.content`, and records `prompt_tokens` / `completion_tokens`
plus a content-addressed receipt. It does not print or persist the key.

## Setup (recommended: file outside the repo)

Create a private file outside the repository, for example
`~/.config/network_evolve/deepseek.env`:

```bash
mkdir -p ~/.config/network_evolve
chmod 700 ~/.config/network_evolve
cat > ~/.config/network_evolve/deepseek.env <<'ENV'
export NETWORK_EVOLVE_LLM_BASE_URL="https://api.deepseek.com/v1"
export NETWORK_EVOLVE_LLM_API_KEY="sk-REPLACE_WITH_YOUR_KEY"
export NETWORK_EVOLVE_LLM_API_STYLE="openai"
ENV
chmod 600 ~/.config/network_evolve/deepseek.env
```

Then load it in any shell that runs an experiment:

```bash
set -a
. ~/.config/network_evolve/deepseek.env
set +a
```

Do **not** put the key in `.env` inside the repository, in a shell command
committed to git, or in the chat. `git status` must remain clean of any key.

## One-call smoke test

```bash
set -a; . ~/.config/network_evolve/deepseek.env; set +a
python - <<'PY'
from src.network_evolve.llm_backend import LLMClient
c = LLMClient(model="deepseek-chat", budget_calls=2, max_tokens=256,
              thinking_budget=0)
text, receipt = c.complete(
    "You return only JSON.",
    'Return exactly {"ok": true} as a JSON object.',
)
print("api_style:", c.api_style)
print("response:", text[:200])
print("tokens:", receipt.input_tokens, receipt.output_tokens)
print("archive:", c.archive.count("deepseek-chat"))
PY
```

The receipt is stored under `data/llm_archive/deepseek-chat/` and is covered by
the provenance manifest when the experiment is recorded.

## Full high-coverage second-model arm

```bash
set -a; . ~/.config/network_evolve/deepseek.env; set +a
python scripts/run_phase6_parallel.py \
  --seeds 10 --jobs 5 --rounds 30 --nodes 60 --llm-events 60 \
  --model deepseek-chat --max-tokens 8192 --thinking-budget 0 \
  --budget-calls 400 \
  --out results/phase6_highcov_60_deepseek.json \
  --shard-dir results/phase6_highcov_60_deepseek_shards
```

This makes 300 batched calls. DeepSeek's rate limits/cost are provider-side;
the script shards by seed and writes one JSON per seed, then merges the paired
statistics. Do not use `deepseek-reasoner` for the main arm without checking
that its response JSON is not truncated; `deepseek-chat` is the
non-thinking-compatible choice.

## What must be updated after the run

1. add the new result to `scripts/record_provenance.py`;
2. add a model row to `paper/tables/llm_highcov.tex` / `llm_models` table;
3. update the paper limitation text: it currently says no second independent
   model family was available. Once the DeepSeek arm exists, that sentence
   must be replaced with the actual comparison and its caveats;
4. run `record_provenance.py`, `check_paper_integrity.py`,
   `generate_paper_tables.py --check`, tests, and compile.
