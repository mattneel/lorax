#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MIX_ENV="${MIX_ENV:-test}"

source "${ROOT_DIR}/scripts/exla_cuda_env.sh"
export INSTALL_MISSING_CUDA_LIBS=1
configure_exla_cuda_env

echo "==> Rebuilding XLA/EXLA with XLA_TARGET=${XLA_TARGET} (MIX_ENV=${MIX_ENV})"
echo "==> Using CUDA_LIB_DIR=${CUDA_LIB_DIR}"
echo "==> Using NCCL_LIB_DIR=${NCCL_LIB_DIR}"
mix deps.clean xla --build
mix deps.clean exla --build
mix deps.get
MIX_ENV="${MIX_ENV}" mix deps.compile xla exla

echo "==> Verifying EXLA platforms"
MIX_ENV="${MIX_ENV}" mix run -e '
Application.ensure_all_started(:exla)
platforms = EXLA.Client.get_supported_platforms()
IO.inspect(platforms, label: "supported_platforms")
if Map.get(platforms, :cuda, 0) < 1 do
  raise "CUDA platform unavailable in EXLA. Check XLA_TARGET/CUDA/cuDNN setup."
end
client_name = EXLA.Client.default_name()
client = EXLA.Client.fetch!(client_name)
IO.inspect(%{
  default_client: client_name,
  platform: client.platform,
  device_count: client.device_count,
  default_device_id: client.default_device_id
}, label: "default_client")
'

echo "==> EXLA CUDA setup complete"
