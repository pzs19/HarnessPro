#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 JOB_NAME JOBS_DIR TASK_PATH" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
job_name="$1"
jobs_dir="$2"
task_path="$3"
codex_binary="${HARNESSPRO_CODEX_BINARY:-$(command -v codex || true)}"
container_base_url="${RHO_PROGRAMBENCH_CONTAINER_BASE_URL:-}"
if [[ -n "${RHO_PROGRAMBENCH_AGENT_API_KEY:-}" ]]; then
  agent_key_template='OPENAI_API_KEY=${RHO_PROGRAMBENCH_AGENT_API_KEY}'
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  agent_key_template='OPENAI_API_KEY=${OPENAI_API_KEY}'
else
  agent_key_template=''
fi
pier_binary="${HARNESSPRO_PIER_BINARY:-$repo_root/.venv/bin/pier}"
model="${RHO_PROGRAMBENCH_MODEL:-gpt-5.5}"
reasoning_effort="${RHO_PROGRAMBENCH_REASONING_EFFORT:-high}"
trace_id="${RHO_PROGRAMBENCH_TRACE_ID:-$job_name}"
harness_dir="${RHO_PROGRAMBENCH_HARNESS_DIR:-}"
runtime_config="${RHO_PROGRAMBENCH_RUNTIME_CONFIG:-}"
runtime_selection="${RHO_PROGRAMBENCH_RUNTIME_SELECTION:-}"

if [[ ! -x "$codex_binary" ]]; then
  echo "Set HARNESSPRO_CODEX_BINARY to an executable Linux Codex binary" >&2
  exit 2
fi
codex_binary="$(readlink -f "$codex_binary")"
if [[ ! -x "$pier_binary" ]]; then
  echo "Pier is missing; run uv sync in the release directory" >&2
  exit 2
fi
if [[ -z "$container_base_url" || -z "$agent_key_template" ]]; then
  echo "Set RHO_PROGRAMBENCH_CONTAINER_BASE_URL and RHO_PROGRAMBENCH_AGENT_API_KEY (or OPENAI_API_KEY)" >&2
  exit 2
fi

provider_config=$(printf '%s\n' 'model_provider = "rho"')
if [[ -n "$runtime_config" ]]; then
  if [[ ! -f "$runtime_config" ]]; then
    echo "ProgramBench runtime config is missing: $runtime_config" >&2
    exit 2
  fi
  provider_config+=$'\n'
  provider_config+="$(cat "$runtime_config")"
elif [[ -n "$harness_dir" ]]; then
  if [[ ! -f "$harness_dir/README.md" ]]; then
    echo "ProgramBench Harness README is missing: $harness_dir/README.md" >&2
    exit 2
  fi
  provider_config+=$'\nmodel_instructions_file = "/harness/README.md"'
fi
provider_config+=$(printf '\n%s\n' \
  '[model_providers.rho]' \
  'name = "HarnessPro Responses"' \
  "base_url = \"$container_base_url\"" \
  'env_key = "OPENAI_API_KEY"' \
  'env_http_headers = { "X-Rho-Transport-Trace" = "RHO_TRANSPORT_TRACE_ID" }' \
  'wire_api = "responses"' \
  'supports_websockets = false')

if [[ -n "$runtime_selection" ]]; then
  if [[ -z "$harness_dir" || ! -f "$runtime_selection" ]]; then
    echo "ProgramBench runtime selection or harness is missing" >&2
    exit 2
  fi
  mounts_json=$(printf \
    '[{"type":"bind","source":"%s","target":"/usr/local/bin/codex","read_only":true},{"type":"bind","source":"%s","target":"/harness","read_only":true},{"type":"bind","source":"%s","target":"/rho-runtime/selection.json","read_only":true}]' \
    "$codex_binary" "$harness_dir" "$runtime_selection")
elif [[ -n "$harness_dir" ]]; then
  mounts_json=$(printf \
    '[{"type":"bind","source":"%s","target":"/usr/local/bin/codex","read_only":true},{"type":"bind","source":"%s","target":"/harness","read_only":true}]' \
    "$codex_binary" "$harness_dir")
else
  mounts_json=$(printf \
    '[{"type":"bind","source":"%s","target":"/usr/local/bin/codex","read_only":true}]' \
    "$codex_binary")
fi

exec "$pier_binary" run \
  --job-name "$job_name" \
  --jobs-dir "$jobs_dir" \
  --path "$task_path" \
  --agent-import-path rho.datasets.programbench.agents:MountedProgramBenchCodex \
  --model "openai/$model" \
  --agent-kwarg "command_model_name=$model" \
  --agent-kwarg "reasoning_effort=$reasoning_effort" \
  --agent-kwarg "config_toml=$provider_config" \
  --agent-env "$agent_key_template" \
  --agent-env "RHO_TRANSPORT_TRACE_ID=$trace_id" \
  --agent-env "RHO_RUNTIME_SELECTION_PATH=/rho-runtime/selection.json" \
  --environment-import-path rho.datasets.programbench.environment:MountedBinaryDockerEnvironment \
  --mounts-json "$mounts_json" \
  --override-cpus "${RHO_PROGRAMBENCH_CPUS:-8}" \
  --override-memory-mb "${RHO_PROGRAMBENCH_MEMORY_MB:-32768}" \
  --n-concurrent 1 \
  --max-retries 0 \
  --disable-verification \
  --delete \
  --debug \
  --yes
