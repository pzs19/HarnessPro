# Evaluation guide

Run the commands below from the repository root. For an overview, see the
[project README](../README.md).

## Setup and release scope

Use Linux with Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), Docker and
Docker Compose, and a Codex CLI executable. The ProgramBench container requires
a Linux-native Codex binary; Codex CLI 0.147.0 is the version used in our runs.
Install the Python dependencies with:

```bash
uv sync --frozen
uv run python scripts/prepare_runtime.py
```

This creates a local `.venv`; the commands below use it through `uv run`.
The preparation command restores executable permissions on bundled runtime
tools and verification helpers, including after a ZIP download.
The bundle includes nine harnesses, the three evaluation runners, their helper
scripts and Python modules, task-ID manifests, and the ProgramBench/Pier evaluator
source. It does **not** include benchmark data, task or grader Docker images,
model weights or services, API endpoints, or credentials. Configure and obtain
those separately before running an evaluation.
The default RTK 0.28.2 helper is bundled for Linux x86-64; on another platform,
provide a compatible executable with `--rtk-binary` for Terminal-Bench 2 and
SWE-bench Pro.

The package's code is licensed under [MIT](../LICENSE). Bundled evaluators retain
their own licenses and notices; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Benchmark inputs are separate: provide Terminal-Bench 2 task directories and
the SWE-bench Pro test Parquet (`ScaleAI/SWE-bench_Pro`), plus any required Docker
images. ProgramBench's task metadata is included, while its hidden grader
archives and task images are fetched in the preparation step below. The three
test-ID manifests used for these evaluations are in `splits/`.

Set `CODEX_CONFIG` to your Codex Responses-provider TOML and `CODEX_BIN` to
your Codex executable. Keep credentials in your local environment.

## Terminal-Bench 2

Set `TB2_TASKS` to the directory containing the benchmark tasks.

```bash
uv run python scripts/run_tb2_harness_eval.py \
  --dataset "$TB2_TASKS" --heldout-ids splits/terminal-bench-2-test.txt \
  --run-dir runs/tb2-medium --harness-label medium \
  --harness-dir harnesses/terminal-bench-2/medium \
  --codex-config "$CODEX_CONFIG" --codex-binary "$CODEX_BIN" \
  --model gpt-5.5 --concurrency 4
```

## SWE-bench Pro

For SWE-bench Pro, set `SWE_PRO_PARQUET` to the downloaded test Parquet. Go tasks
need the exact Go releases reported by their official evaluator images. Before
evaluating, inspect the required versions (this may pull missing task images),
then install each official Go distribution under
`third_party/toolchains/go<version>/bin/go` or set `RHO_GO_TOOLCHAIN_ROOT` to an
equivalent directory. The Go distributions are not bundled.

```bash
uv run python scripts/inspect_swebench_pro_go_toolchains.py \
  --dataset "$SWE_PRO_PARQUET" --heldout-ids splits/swe-bench-pro-test.txt \
  --docker-executable "$(command -v docker)"
```

The evaluator prepares its repository cache and official Docker assets in the
selected cache root. Put the bundled Go wrapper on the Solver's `PATH` so a
missing exact release fails visibly instead of silently using system Go:

```bash
uv run python scripts/run_swebench_pro_harness_eval.py \
  --dataset "$SWE_PRO_PARQUET" --heldout-ids splits/swe-bench-pro-test.txt \
  --run-dir runs/swe-pro-medium --harness-label medium \
  --harness-dir harnesses/swe-bench-pro/medium \
  --codex-config "$CODEX_CONFIG" --codex-binary "$CODEX_BIN" \
  --docker-command "$(command -v docker)" --cache-root cache/swe-bench-pro \
  --solver-env "PATH=$(pwd)/scripts/swebench-pro-toolchain-bin:$PATH" \
  --model gpt-5.5 --solver-timeout-s 1200 --concurrency 10
```

## ProgramBench

For ProgramBench, set `HARNESSPRO_CODEX_BINARY` to a Linux-native Codex binary,
`RHO_PROGRAMBENCH_CONTAINER_BASE_URL` to a Responses endpoint reachable **from
the task container**, and `RHO_PROGRAMBENCH_AGENT_API_KEY` in your environment.
Prepare the official task images and hidden-test archives before starting model
calls:

```bash
uv run python scripts/prefetch_programbench.py \
  --task-ids splits/programbench-test.txt \
  --docker-executable "$(command -v docker)" \
  --output cache/programbench-preflight.json
uv run python scripts/generate_programbench_tasks.py \
  --programbench-root third_party/programbench \
  --task-ids splits/programbench-test.txt \
  --output-root cache/programbench-tasks
uv run python scripts/run_programbench_eval.py \
  --task-ids splits/programbench-test.txt \
  --tasks-root cache/programbench-tasks \
  --programbench-root third_party/programbench \
  --docker-executable "$(command -v docker)" \
  --run-dir runs/programbench-medium \
  --harness-dir harnesses/programbench/medium \
  --model gpt-5.5 --concurrency 5
```

## Runtime notes

Use a fresh `--run-dir` for each benchmark and mode. All three runners accept
`--help` and `--max-tasks 1` for a one-task smoke test. ProgramBench's API key
is read from the environment by Pier at runtime; keep it out of shell history
and shared logs.

The released evaluation entry points support Codex and Claude Code on
Terminal-Bench 2 and SWE-bench Pro. ProgramBench's supplied runner uses Codex.
See [Claude Code setup](claude-code.md) for the container build, native CLI
setup, and explicit endpoint configuration. SWE-bench Pro defaults to a
1200-second solver limit for GPT-5.5 and 7200 seconds for `--model GLM-5.2`.
The paper's GLM-5.2 evolved harness snapshots are not included in this release.
Using the GPT-5.5 harness files with the Claude Code backend does not reproduce
those GLM-5.2 results.

## Harness design

These harnesses are more than alternative prompts. Depending on the benchmark and
mode, they combine model instructions with task-adaptive reasoning selection,
staged tool-use guidance, output compaction, and verification helpers. Some modes
use a fixed profile instead of a classifier; the `runtime/policy.json` in each
directory is the source of truth for its behavior.

The controls are tuned at the benchmark level, while profile selection uses only
the public task description. No hidden tests, previous benchmark outcomes, or
task-specific fingerprints are needed to choose a profile.

## Repository layout

```text
HarnessPro/
├── README.md, pyproject.toml, uv.lock
├── docs/                       # evaluation and backend setup
├── harnesses/
│   ├── terminal-bench-2/{lite,medium,heavy}/
│   ├── swe-bench-pro/{lite,medium,heavy}/
│   └── programbench/{lite,medium,heavy}/
├── scripts/                    # evaluation runners and preparation tools
├── splits/                     # evaluation task ID manifests
├── src/rho/                    # harness runtime and benchmark adapters
└── third_party/                # ProgramBench and Pier evaluator source
```

Each `harnesses/<benchmark>/<mode>/` contains its task-facing `README.md`
and `runtime/` policy, instructions, and hook. Some modes also contain a
verification helper or checklist. Generated runs, caches, Docker images,
benchmark task inputs, and credentials are not part of this package.

## Offline checks

```bash
uv run python -B -m unittest discover -s tests -v
```

These checks cover configuration loading, model-specific SWE-bench Pro timeouts,
supported backend selection, and Claude Code hook compatibility. They do not
make model requests or launch benchmark containers.
