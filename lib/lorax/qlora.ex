defmodule Lorax.QLoRA do
  @moduledoc """
  Validation helpers for QLoRA target configuration.
  """

  alias Lorax.QLoRA.Provider
  alias Lorax.QLoRA.Target

  @supported_ops [:dense, :conv]

  @doc """
  Validates QLoRA targets against the selected node ids.

  Returns normalized `%Lorax.QLoRA.Target{}` structs keyed by target layer name.
  """
  def validate_targets!(%Axon{}, _target_ids, nil), do: %{}

  def validate_targets!(%Axon{} = axon, target_ids, qlora_targets)
      when is_map(qlora_targets) and map_size(qlora_targets) == 0 do
    validate_target_ops!(axon, target_ids)
    %{}
  end

  def validate_targets!(%Axon{} = axon, target_ids, qlora_targets) when is_map(qlora_targets) do
    target_name_to_node = target_name_node_map(axon, target_ids)
    selected_target_names = Map.keys(target_name_to_node)
    config_target_names = Map.keys(qlora_targets)

    missing_names = selected_target_names -- config_target_names

    if missing_names != [] do
      missing_names =
        missing_names
        |> Enum.sort()
        |> Enum.join(", ")

      raise ArgumentError,
            "missing QLoRA targets for selected layers: #{missing_names}"
    end

    unknown_names = config_target_names -- selected_target_names

    if unknown_names != [] do
      unknown_names =
        unknown_names
        |> Enum.sort()
        |> Enum.join(", ")

      raise ArgumentError,
            "unknown QLoRA target layer names: #{unknown_names}"
    end

    Enum.reduce(target_name_to_node, %{}, fn {target_name, node}, acc ->
      validate_target_op!(target_name, node)

      target =
        qlora_targets
        |> Map.fetch!(target_name)
        |> normalize_target!(target_name)

      Map.put(acc, target_name, target)
    end)
  end

  def validate_targets!(%Axon{}, _target_ids, qlora_targets) do
    raise ArgumentError,
          "`qlora_targets` must be a map keyed by target layer name, got: #{inspect(qlora_targets)}"
  end

  defp validate_target_ops!(axon, target_ids) do
    Enum.each(target_ids, fn target_id ->
      node = Map.fetch!(axon.nodes, target_id)
      target_name = target_name(node)
      validate_target_op!(target_name, node)
    end)
  end

  defp validate_target_op!(_target_name, %Axon.Node{op: op}) when op in @supported_ops do
    :ok
  end

  defp validate_target_op!(target_name, %Axon.Node{op: op}) do
    raise ArgumentError,
          "QLoRA only supports target ops #{inspect(@supported_ops)}, got #{inspect(op)} for layer #{target_name}"
  end

  defp target_name_node_map(%Axon{} = axon, target_ids) do
    Enum.reduce(target_ids, %{}, fn target_id, acc ->
      node = Map.fetch!(axon.nodes, target_id)
      Map.put(acc, target_name(node), node)
    end)
  end

  defp normalize_target!(%Target{} = target, target_name) do
    target
    |> maybe_default_provider_opts()
    |> validate_target_struct!(target_name)
  end

  defp normalize_target!(%{} = target_map, target_name) do
    target_map
    |> then(&struct(Target, &1))
    |> maybe_default_provider_opts()
    |> validate_target_struct!(target_name)
  end

  defp normalize_target!(target, target_name) do
    raise ArgumentError,
          "QLoRA target for layer #{target_name} must be a map or %Lorax.QLoRA.Target{}, got: #{inspect(target)}"
  end

  defp maybe_default_provider_opts(%Target{provider_opts: nil} = target) do
    %Target{target | provider_opts: %{}}
  end

  defp maybe_default_provider_opts(%Target{} = target), do: target

  defp validate_target_struct!(
         %Target{
           packed_weight: %Nx.Tensor{},
           provider: provider,
           provider_opts: provider_opts,
           base_bias: base_bias,
           base_type: base_type
         } = target,
         target_name
       )
       when is_atom(provider) do
    if not (is_map(provider_opts) or is_list(provider_opts)) do
      raise ArgumentError,
            "`provider_opts` for QLoRA layer #{target_name} must be a map or keyword list"
    end

    if not is_tuple(base_type) do
      raise ArgumentError,
            "`base_type` for QLoRA layer #{target_name} must be an Nx type tuple, got: #{inspect(base_type)}"
    end

    if not is_nil(base_bias) and not match?(%Nx.Tensor{}, base_bias) do
      raise ArgumentError,
            "`base_bias` for QLoRA layer #{target_name} must be an Nx tensor or nil"
    end

    ensure_provider!(provider, target_name)
    target
  end

  defp validate_target_struct!(%Target{} = target, target_name) do
    raise ArgumentError,
          "QLoRA target for layer #{target_name} is invalid, got: #{inspect(target)}"
  end

  defp ensure_provider!(provider, target_name) do
    with true <- Code.ensure_loaded?(provider),
         true <- function_exported?(provider, :dequant, 2),
         true <- provider_behaviour?(provider) do
      :ok
    else
      _ ->
        raise ArgumentError,
              "QLoRA provider #{inspect(provider)} for layer #{target_name} does not implement #{inspect(Provider)}"
    end
  end

  defp provider_behaviour?(provider) do
    behaviours =
      provider.module_info(:attributes)
      |> Keyword.get(:behaviour, [])

    Provider in behaviours
  end

  defp target_name(%Axon.Node{name: name_fn}) do
    name_fn.(nil, nil)
  end
end
