#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/exla_cuda_env.sh"
configure_exla_cuda_env

export LORAX_TEST_EXLA=1

exec mix test --only gpu "$@"
