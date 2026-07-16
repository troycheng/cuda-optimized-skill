#!/usr/bin/env bash
set -euo pipefail

lane="${1:-compat}"
case "$lane" in
  current)
    requested_ref="${CUDA_CURRENT_IMAGE:-cuda-skill-current:cuda13.3-triton3.7.1-ncu2026.2.1}"
    ;;
  compat)
    requested_ref="${CUDA_COMPAT_IMAGE:-lmsysorg/sglang:latest-cu130-runtime}"
    ;;
  *)
    echo "usage: $0 [current|compat]" >&2
    exit 2
    ;;
esac

readonly allowed_root="/data/tcheng/cuda-skill-e2e/v2.2"
readonly allowed_cutlass="/data/tcheng/cuda-skill-e2e/deps/cutlass"
readonly expected_cutlass_version="4.6.1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../../../.." && pwd -P)"
if [[ "$repo_root" != "$allowed_root/repo" ]]; then
  echo "refusing repository outside $allowed_root/repo: $repo_root" >&2
  exit 2
fi

: "${CUDA_E2E_ARTIFACTS:?set CUDA_E2E_ARTIFACTS below $allowed_root/artifacts}"
: "${CUTLASS_PATH:?set CUTLASS_PATH to $allowed_cutlass}"
artifacts="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "$CUDA_E2E_ARTIFACTS")"
case "$artifacts/" in
  "$allowed_root/artifacts/"*) ;;
  *)
    echo "refusing artifact path outside $allowed_root/artifacts: $artifacts" >&2
    exit 2
    ;;
esac
mkdir -p -- "$artifacts"
artifacts="$(cd -- "$artifacts" && pwd -P)"
case "$artifacts/" in
  "$allowed_root/artifacts/"*) ;;
  *)
    echo "refusing resolved artifact path outside $allowed_root/artifacts: $artifacts" >&2
    exit 2
    ;;
esac

shopt -s nullglob dotglob
for entry in "$artifacts"/*; do
  if [[ "$(basename -- "$entry")" == "run.log" && -f "$entry" && ! -L "$entry" ]]; then
    continue
  fi
  echo "artifact lane must be empty except for a regular run.log: $artifacts" >&2
  exit 2
done
shopt -u nullglob dotglob

case "$CUTLASS_PATH" in
  *vllm-opt*)
    echo "refusing CUTLASS path overlapping vllm-opt: $CUTLASS_PATH" >&2
    exit 2
    ;;
esac
cutlass_path="$(cd -- "$CUTLASS_PATH" && pwd -P)"
if [[ "$cutlass_path" != "$allowed_cutlass" ]]; then
  echo "CUTLASS physical path must equal $allowed_cutlass: $cutlass_path" >&2
  exit 2
fi
case "$cutlass_path/" in
  "$repo_root/"*)
    echo "CUTLASS checkout must not overlap the test repository" >&2
    exit 2
    ;;
esac
case "$repo_root/" in
  "$cutlass_path/"*)
    echo "test repository must not overlap the CUTLASS checkout" >&2
    exit 2
    ;;
esac
for required in \
  "$cutlass_path/include/cutlass/cutlass.h" \
  "$cutlass_path/include/cutlass/version.h"; do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "required CUTLASS file is missing or a symlink: $required" >&2
    exit 2
  fi
done
cutlass_version="$(python3 - "$cutlass_path/include/cutlass/version.h" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
components = []
for name in ("MAJOR", "MINOR", "PATCH"):
    matches = re.findall(
        rf"^\s*#\s*define\s+CUTLASS_{name}\s+(\d+)\s*(?://.*)?$",
        source,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise SystemExit(f"CUTLASS_{name} must have one numeric definition")
    components.append(matches[0])
print(".".join(components))
PY
)"
if [[ "$cutlass_version" != "$expected_cutlass_version" ]]; then
  echo "CUTLASS version must be $expected_cutlass_version: $cutlass_version" >&2
  exit 2
fi

gpu="${CUDA_E2E_GPU:-1}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "CUDA_E2E_GPU must be a non-negative GPU index" >&2
  exit 2
fi
gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)"

assert_gpu_idle() {
  local processes busy
  processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)"
  busy="$(printf '%s\n' "$processes" \
    | awk -F, -v uuid="$gpu_uuid" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)} $1 == uuid {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')"
  if [[ -n "$busy" ]]; then
    echo "refusing busy GPU $gpu ($gpu_uuid), compute PIDs: $busy" >&2
    exit 3
  fi
}

assert_gpu_idle

resolved_image_id="$(docker image inspect --format '{{.Id}}' "$requested_ref")"
if [[ ! "$resolved_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image did not resolve to an immutable sha256 ID: $requested_ref" >&2
  exit 2
fi
verified_image_id="$(docker image inspect --format '{{.Id}}' "$resolved_image_id")"
if [[ "$verified_image_id" != "$resolved_image_id" ]]; then
  echo "immutable image identity changed during inspection" >&2
  exit 2
fi
repo_digests="$(docker image inspect --format '{{json .RepoDigests}}' "$resolved_image_id")"
created="$(docker image inspect --format '{{.Created}}' "$resolved_image_id")"
image_os="$(docker image inspect --format '{{.Os}}' "$resolved_image_id")"
architecture="$(docker image inspect --format '{{.Architecture}}' "$resolved_image_id")"
python3 - \
  "$artifacts/container-image.json" \
  "$requested_ref" \
  "$resolved_image_id" \
  "$repo_digests" \
  "$created" \
  "$image_os" \
  "$architecture" <<'PY'
import json
import pathlib
import sys

path, requested_ref, resolved_id, repo_digests, created, image_os, architecture = sys.argv[1:]
payload = {
    "requested_ref": requested_ref,
    "resolved_id": resolved_id,
    "repo_digests": json.loads(repo_digests),
    "created": created,
    "os": image_os,
    "architecture": architecture,
}
pathlib.Path(path).write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

mkdir -p -- "$artifacts/home" "$artifacts/triton-cache" "$artifacts/pycache"
assert_gpu_idle

exec docker run --rm \
  --pull never \
  --gpus "device=$gpu" \
  --network none \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --ipc private \
  --user "$(id -u):$(id -g)" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CUDA_SM120_E2E=1 \
  -e CUDA_E2E_ARTIFACTS="$artifacts" \
  -e CUTLASS_PATH="$cutlass_path" \
  -e HOME="$artifacts/home" \
  -e TRITON_CACHE_DIR="$artifacts/triton-cache" \
  -e PYTHONPYCACHEPREFIX="$artifacts/pycache" \
  -v "$repo_root:$repo_root:ro" \
  -v "$artifacts:$artifacts:rw" \
  -v "$cutlass_path:$cutlass_path:ro" \
  -w "$repo_root" \
  "$resolved_image_id" \
  python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
