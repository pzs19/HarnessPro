# Claude Code evaluation setup

The Terminal-Bench 2 and SWE-bench Pro runners support `--agent-backend
claude-code`. Terminal-Bench 2 runs the CLI in a container; SWE-bench Pro runs
it natively so the solver can use the host repository cache and Go toolchains.
The ProgramBench runner supplied in this release uses Codex.

These are backend setup instructions. The published harness directories are
the GPT-5.5/Codex snapshots. The GLM-5.2 evolved snapshots used in the paper are
not included, so changing the model or backend alone does not reproduce the
GLM-5.2 results. Use the corresponding experiment snapshots when available.

## Shared configuration

Run `uv sync --frozen` and `uv run python scripts/prepare_runtime.py` first.
Provide an Anthropic Messages-compatible gateway for the desired model through
`HARNESSPRO_CLAUDE_BASE_URL`, and set `HARNESSPRO_CLAUDE_API_KEY` in your local
environment. Neither the gateway nor its upstream provider credentials is
bundled. The runners no longer assume an experiment-local router or port.

`--claude-code-ccr-url` and `--claude-code-client-key` override those environment
variables. Use the environment for credentials. Set
`--claude-code-model-alias` to the model name configured in your gateway.
The default alias is `sonnet`; it is a gateway routing name, not evidence of
which upstream model served a request. Confirm routing on the gateway.

The task-profile classifier is a separate model request. The provided GPT-5.5
policies use a Responses transport and need the `CODEX_CONFIG` described in the
[evaluation guide](evaluation.md#setup-and-release-scope). Switching the solver
to Claude Code does not switch that classifier's transport, model, or rubric.
GLM-specific configurations must
supply their own classifier settings and corresponding credentials.

For cost measurements, the adapter can consume router accounting using
`--claude-code-ccr-metrics-url` together with `--claude-code-trace-usage`.
That accounting service is external to this release. A working backend launch
does not by itself validate provider usage or reproduce the paper's costs.

## Terminal-Bench 2 container

Build the solver image from the supplied Dockerfile:

```bash
docker build -f docker/claude-code.Dockerfile \
  -t harnesspro-claude-code:2.1.231 .
docker run --rm harnesspro-claude-code:2.1.231 --version
```

The image installs the pinned official `@anthropic-ai/claude-code` npm package
and Python 3 for the runtime hooks. It runs as an unprivileged user. A local
Docker daemon and a Linux-native Docker client are also required: the runner
mounts that client and the daemon socket so the solver can operate the task
container. Set `DOCKER_CLIENT` to the actual Linux Docker executable. A shell
wrapper or a Snap launcher is not the executable to mount. Set `DOCKER_SOCKET`
to the local Unix socket, usually `/var/run/docker.sock`; rootless installations
must supply their own socket path.

The gateway URL must be reachable from the solver container. A gateway on the
host can use `host.docker.internal` with its explicitly configured port. The
runner adds the host-gateway mapping and the socket/workspace supplementary
groups. It does not require the old `scripts/tb2-infra/` wrapper.

For a backend integration check with a supplied GPT-5.5 harness:

```bash
uv run python scripts/run_tb2_harness_eval.py \
  --dataset "$TB2_TASKS" --heldout-ids splits/terminal-bench-2-test.txt \
  --run-dir runs/tb2-claude-backend-check --harness-label backend-check \
  --harness-dir harnesses/terminal-bench-2/medium \
  --codex-config "$CODEX_CONFIG" \
  --agent-backend claude-code --model "$SOLVER_MODEL" \
  --claude-code-model-alias "$CLAUDE_MODEL_ALIAS" \
  --claude-code-docker-command "$(command -v docker)" \
  --claude-code-docker-client "$DOCKER_CLIENT" \
  --claude-code-docker-socket "$DOCKER_SOCKET" \
  --concurrency 1 --max-tasks 1
```

Set `SOLVER_MODEL` to the actual upstream model and `CLAUDE_MODEL_ALIAS` to the
gateway alias. With runtime profile routing enabled, this runner appends the
selected reasoning effort to the router client key, for example `-medium` or
`-high`. The gateway must accept and route those client keys. The base client
key by itself is used for static harnesses. This router convention is part of
the existing adapter, not a feature supplied by the Docker image.

## SWE-bench Pro native CLI

Install the pinned Claude Code CLI in an environment with Node.js 22 or newer:

```bash
npm install -g @anthropic-ai/claude-code@2.1.231
claude --version
```

Prepare benchmark inputs, Docker grading images, repository caches, and exact
Go toolchains as described in the [evaluation guide](evaluation.md#swe-bench-pro).
Then supply the installed native CLI with `--claude-code-binary`:

```bash
uv run python scripts/run_swebench_pro_harness_eval.py \
  --dataset "$SWE_PRO_PARQUET" --heldout-ids splits/swe-bench-pro-test.txt \
  --run-dir runs/swe-pro-claude-backend-check --harness-label backend-check \
  --harness-dir harnesses/swe-bench-pro/medium \
  --codex-config "$CODEX_CONFIG" \
  --docker-command "$(command -v docker)" --cache-root cache/swe-bench-pro \
  --solver-env "PATH=$(pwd)/scripts/swebench-pro-toolchain-bin:$PATH" \
  --agent-backend claude-code --model "$SOLVER_MODEL" \
  --claude-code-binary "$(command -v claude)" \
  --claude-code-model-alias "$CLAUDE_MODEL_ALIAS" \
  --solver-timeout-s "$SOLVER_TIMEOUT_S" --concurrency 1 --max-tasks 1
```

For the paper's SWE-bench Pro budgets, use `SOLVER_TIMEOUT_S=1200` for GPT-5.5
and `SOLVER_TIMEOUT_S=7200` for GLM-5.2. When the timeout option is omitted,
the runner selects 7200 seconds for `--model GLM-5.2` (case-insensitive) and
1200 seconds otherwise. Terminal-Bench 2 retains the benchmark's per-task
solver watchdog rather than using the SWE-bench Pro limit.
