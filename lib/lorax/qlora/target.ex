defmodule Lorax.QLoRA.Target do
  @moduledoc """
  QLoRA target specification used by `Lorax.inject/2`.
  """

  @type t :: %__MODULE__{
          packed_weight: Nx.Tensor.t() | nil,
          provider: module() | nil,
          provider_opts: term(),
          base_bias: Nx.Tensor.t() | nil,
          base_type: Nx.Type.t()
        }

  defstruct packed_weight: nil,
            provider: nil,
            provider_opts: %{},
            base_bias: nil,
            base_type: {:f, 32}
end
