defmodule Lorax.QLoRA.Provider do
  @moduledoc """
  Behaviour contract for QLoRA dequantization providers.

  Implementations must keep `dequant/2` compatible with `Nx.Defn` usage.
  In practice this means the function should rely on Nx-compatible operations
  and return an `Nx.Tensor`.
  """

  @callback dequant(packed_weight :: Nx.Tensor.t(), opts :: term()) :: Nx.Tensor.t()
end
