defmodule Lorax.QLoRATest do
  use ExUnit.Case, async: false

  defmodule CastProvider do
    @behaviour Lorax.QLoRA.Provider

    @impl true
    def dequant(%Nx.Tensor{} = packed_weight, opts) do
      type =
        cond do
          is_map(opts) -> Map.get(opts, :type, {:f, 32})
          is_list(opts) -> Keyword.get(opts, :type, {:f, 32})
          true -> {:f, 32}
        end

      Nx.as_type(packed_weight, type)
    end
  end

  defmodule BadProvider do
    def dequant(_packed_weight, _opts), do: Nx.tensor(0.0)
  end

  test "inject/2 fails fast when qlora target entries are missing" do
    model = qkv_dense_model()

    config = %Lorax.Config{
      qlora_targets: %{
        "query" => %Lorax.QLoRA.Target{
          packed_weight: Nx.iota({4, 6}, type: {:u, 8}),
          provider: CastProvider
        }
      }
    }

    assert_raise ArgumentError, ~r/missing QLoRA targets/, fn ->
      Lorax.inject(model, config)
    end
  end

  test "inject/2 fails when provider does not implement behaviour" do
    model = qkv_dense_model()

    config = %Lorax.Config{
      qlora_targets: dense_qlora_targets(BadProvider)
    }

    assert_raise ArgumentError, ~r/does not implement Lorax\.QLoRA\.Provider/, fn ->
      Lorax.inject(model, config)
    end
  end

  test "inject/2 supports dense qlora targets and preserves output shape" do
    model = qkv_dense_model()

    config = %Lorax.Config{
      qlora_targets: dense_qlora_targets(CastProvider)
    }

    qlora_model = Lorax.inject(model, config)

    qlora_layer_names =
      qlora_model
      |> model_node_names()
      |> Enum.filter(&String.starts_with?(&1, "lora_"))
      |> Enum.sort()

    assert qlora_layer_names == ["lora_key", "lora_query", "lora_value"]

    {base_init, base_predict} = Axon.build(model, mode: :train, seed: 7)
    {qlora_init, qlora_predict} = Axon.build(qlora_model, mode: :train, seed: 7)

    input = Nx.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    template = Nx.template({2, 4}, :f32)

    base_state = base_init.(template, Axon.ModelState.empty())
    qlora_state = qlora_init.(template, Axon.ModelState.empty())

    assert Nx.shape(base_predict.(base_state, input).prediction) ==
             Nx.shape(qlora_predict.(qlora_state, input).prediction)

    assert qlora_state.parameters |> Map.keys() |> Enum.sort() ==
             ["lora_key", "lora_query", "lora_value"]

    refute Map.has_key?(qlora_state.data, "query")
    refute Map.has_key?(qlora_state.data, "key")
    refute Map.has_key?(qlora_state.data, "value")
  end

  test "inject/2 supports conv qlora targets and preserves output shape" do
    model =
      Axon.input("image", shape: {nil, 8, 8, 3})
      |> Axon.conv(4, kernel_size: {3, 3}, name: "conv")

    config = %Lorax.Config{
      target_node_fn: fn %Axon.Node{op: op, name: name_fn} ->
        op == :conv and name_fn.(nil, nil) == "conv"
      end,
      qlora_targets: %{
        "conv" => %Lorax.QLoRA.Target{
          packed_weight: Nx.iota({3, 3, 3, 4}, type: {:u, 8}),
          provider: CastProvider,
          provider_opts: %{type: {:f, 32}},
          base_bias: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {4})
        }
      }
    }

    qlora_model = Lorax.inject(model, config)
    assert qlora_model |> model_node_names() |> Enum.member?("lora_conv")

    {base_init, base_predict} = Axon.build(model, mode: :train, seed: 7)
    {qlora_init, qlora_predict} = Axon.build(qlora_model, mode: :train, seed: 7)

    input = Nx.iota({2, 8, 8, 3}, type: :f32)
    template = Nx.template({2, 8, 8, 3}, :f32)

    base_state = base_init.(template, Axon.ModelState.empty())
    qlora_state = qlora_init.(template, Axon.ModelState.empty())

    assert Nx.shape(base_predict.(base_state, input).prediction) ==
             Nx.shape(qlora_predict.(qlora_state, input).prediction)

    assert qlora_state.parameters |> Map.keys() |> Enum.sort() == ["lora_conv"]
    refute Map.has_key?(qlora_state.data, "conv")
  end

  test "qlora training updates only lora adapter parameters" do
    model = qkv_dense_model()

    qlora_model =
      Lorax.inject(model, %Lorax.Config{
        r: 2,
        alpha: 4,
        dropout: 0.0,
        dropout_seed: 123,
        qlora_targets: dense_qlora_targets(CastProvider)
      })

    {init_fn, _predict_fn} = Axon.build(qlora_model, mode: :train, seed: 17)
    initial_state = init_fn.(Nx.template({2, 4}, :f32), Axon.ModelState.empty())
    initial_lora_up = initial_state.data["lora_query"]["lora_up"]

    assert Enum.all?(Map.keys(initial_state.parameters), &String.starts_with?(&1, "lora_"))

    trained_state =
      qlora_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.05))
      |> Axon.Loop.run(training_batches(), initial_state, epochs: 2)

    updated_lora_up = trained_state.data["lora_query"]["lora_up"]

    assert Nx.to_number(Nx.all_close(initial_lora_up, updated_lora_up)) == 0
    assert Enum.all?(Map.keys(trained_state.parameters), &String.starts_with?(&1, "lora_"))
    refute Map.has_key?(trained_state.data, "query")
    refute Map.has_key?(trained_state.data, "key")
    refute Map.has_key?(trained_state.data, "value")
  end

  defp qkv_dense_model do
    Axon.input("input", shape: {nil, 4})
    |> Axon.dense(6, activation: :relu, name: "query")
    |> Axon.dense(6, activation: :relu, name: "key")
    |> Axon.dense(3, name: "value")
  end

  defp dense_qlora_targets(provider) do
    %{
      "query" => %Lorax.QLoRA.Target{
        packed_weight: Nx.iota({4, 6}, type: {:u, 8}),
        provider: provider,
        provider_opts: %{type: {:f, 32}},
        base_bias: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {6})
      },
      "key" => %Lorax.QLoRA.Target{
        packed_weight: Nx.iota({6, 6}, type: {:u, 8}),
        provider: provider,
        provider_opts: %{type: {:f, 32}},
        base_bias: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {6})
      },
      "value" => %Lorax.QLoRA.Target{
        packed_weight: Nx.iota({6, 3}, type: {:u, 8}),
        provider: provider,
        provider_opts: %{type: {:f, 32}},
        base_bias: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {3})
      }
    }
  end

  defp model_node_names(%Axon{} = model) do
    Enum.map(model.nodes, fn {_id, %Axon.Node{name: name_fn}} -> name_fn.(nil, nil) end)
  end

  defp training_batches do
    [
      {
        Nx.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        Nx.tensor([[0.2, 0.8, 0.0], [0.8, 0.1, 0.1]])
      },
      {
        Nx.tensor([[0.5, 0.2, 0.1, 0.3], [0.3, 0.4, 0.2, 0.1]]),
        Nx.tensor([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]])
      }
    ]
  end
end
