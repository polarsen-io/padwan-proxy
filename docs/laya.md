# Local Laya model

`--approvals laya` serves TypeSafe's `POST /systemone` route on the proxy itself and points
the [approval hook](approvals.md) there with `PADWAN_PROXY_APPROVALS_URL`. Verdicts are
advisory: the hook always returns `ask`, whatever the confidence gate.

**Prerequisites:** the `laya` extra (`uv sync --extra laya`, several GB of CUDA torch). The
proxy refuses to start without it.

```bash
uv run padwan-proxy --backend-url https://api.scaleway.ai/v1/ -m glm-5.2 \
  --claude-config ~/.claude-scaleway --approvals laya
```

- Weights load once per worker at startup, followed by a warmup call. Startup logs the device,
  including any CPU fallback.
- First use downloads weights from Hugging Face; set `HF_HUB_OFFLINE=1` once cached.
- Inference is serialized under a thread lock. Separate server processes each load their own
  model.
- Switching back to `--approvals jev` drops the URL.
- To use a proxy running elsewhere (such as the [Docker image](docker.md#laya-variant)), set
  `PADWAN_PROXY_APPROVALS_URL` yourself. `--approvals laya` without `--claude-config` serves
  the route and writes no settings.

## Checkpoints

| `--approval-subfolder` | Use for |
|---|---|
| (default) | English requests |
| `multilingual` | French or mixed-language requests |
| `typed-decisions` | The other bundled checkpoint |

Question schemas and option counts (2–20 for choice/score) are checked before inference.

## Limitations

- Laya reads about 316 tokens of state (user requests, working directory, tool name and
  arguments). The tool call sits at the end, so a cut would hide what is being judged: the
  proxy refuses oversized states with `413` and the hook asks. Expect this on any session with
  more than a few short user turns.
- Laya's choice confidence is an entropy score, not its winning probability. Lowering the gate
  does not establish accuracy.
- Local endpoints set through `PADWAN_PROXY_APPROVALS_URL` always stay advisory; the raw
  verdict is still logged.

## Benchmark

The runner reads `benchmarks/data/laya_approvals.jsonl`, which is not in the repository: 140
synthetic cases, 70 English/French pairs, split by task family into 70 development and 70
test cases. No transcripts are used. The
candidate policy asks three factual questions in one batch and maps the answers in code; it
is an experiment, not the deployed policy.

```bash
# HF_HOME must point at a cache containing these checkpoints; nothing is downloaded.
HF_HOME=/path/to/huggingface just benchmark-laya --device cuda \
  --models convaiinnovations/laya 'convaiinnovations/laya#multilingual' \
  'convaiinnovations/laya#typed-decisions' --output .scratch/laya-benchmark
```

The runner records dataset, question, source, config and checkpoint hashes, versions, raw
answers, device, memory and latency. Thresholds are chosen on development cases before test
inference. Reports include unsafe approvals, false denials, escalation, fit rate,
language/tool slices and Wilson intervals (descriptive only: translated pairs are correlated).
`baseline_production_0_99` is the former automatic policy; `always_ask` is the current hook.
Latency covers the four questions batched together, excluding HTTP and hook overhead.

### Results (September 2026, local GPU)

| Checkpoint | Test approvals (of 70) | Unsafe approvals |
|---|---|---|
| English | 0 | 0 |
| multilingual | 7 | 2 |
| typed-decisions | 4 | 0 |

The former 0.99 gate escalated all 70 test cases on every checkpoint. Four approvals are
insufficient evidence of reliability, so the candidate was not promoted and no production
threshold was lowered.
