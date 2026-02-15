defmodule Lorax.TrainingSmokeTest do
  use ExUnit.Case, async: false

  @moduletag :integration

  defmodule SmokeCastProvider do
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

  test "smoke: injected model trains and updates LoRA adapter weights" do
    model = qkv_dense_model()

    lora_model =
      Lorax.inject(model, %Lorax.Config{
        r: 2,
        alpha: 4,
        dropout: 0.0,
        dropout_seed: 123
      })

    template = Nx.template({2, 4}, :f32)

    {base_init, _} = Axon.build(model, mode: :train, seed: 17)
    {lora_init, _} = Axon.build(lora_model, mode: :train, seed: 17)

    base_state = base_init.(template, Axon.ModelState.empty())
    initial_state = lora_init.(template, Axon.ModelState.empty())
    initial_lora_up = initial_state.data["lora_query"]["lora_up"]

    trained_state =
      lora_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.05))
      |> Axon.Loop.run(training_batches(), initial_state, epochs: 2)

    updated_lora_up = trained_state.data["lora_query"]["lora_up"]

    assert Nx.to_number(Nx.all_close(initial_lora_up, updated_lora_up)) == 0

    lora_only_params = Lorax.Params.filter(trained_state.data, base_state.data)

    assert Map.keys(lora_only_params) |> Enum.sort() == ["lora_key", "lora_query", "lora_value"]
  end

  @tag :gpu
  @tag timeout: 180_000
  test "gpu smoke (opt-in): one EXLA training epoch runs end-to-end" do
    exla = :"Elixir.EXLA"
    gpu_smoke? = System.get_env("LORAX_TEST_EXLA") in ["1", "true", "TRUE", "yes", "YES"]

    if gpu_smoke? and Code.ensure_loaded?(exla) do
      case Application.ensure_all_started(:exla) do
        {:ok, _} -> :ok
        {:error, reason} -> flunk("EXLA failed to start: #{inspect(reason)}")
      end

      model = qkv_dense_model()

      lora_model =
        Lorax.inject(model, %Lorax.Config{
          r: 2,
          alpha: 4,
          dropout: 0.0,
          dropout_seed: 123
        })

      {init_fn, _predict_fn} = Axon.build(lora_model, mode: :train, seed: 17, compiler: exla)
      init_state = init_fn.(Nx.template({2, 4}, :f32), Axon.ModelState.empty())

      trained_state =
        lora_model
        |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.sgd(learning_rate: 0.01))
        |> Axon.Loop.run(training_batches(), init_state, epochs: 1, compiler: exla)

      assert match?(%Axon.ModelState{}, trained_state)
    else
      assert true
    end
  end

  @tag :gpu
  @tag timeout: 180_000
  test "gpu smoke (opt-in): one EXLA QLoRA training epoch runs end-to-end" do
    exla = :"Elixir.EXLA"
    gpu_smoke? = System.get_env("LORAX_TEST_EXLA") in ["1", "true", "TRUE", "yes", "YES"]

    if gpu_smoke? and Code.ensure_loaded?(exla) do
      case Application.ensure_all_started(:exla) do
        {:ok, _} -> :ok
        {:error, reason} -> flunk("EXLA failed to start: #{inspect(reason)}")
      end

      model = qkv_dense_model()

      qlora_model =
        Lorax.inject(model, %Lorax.Config{
          r: 2,
          alpha: 4,
          dropout: 0.0,
          dropout_seed: 123,
          qlora_targets: dense_qlora_targets(SmokeCastProvider)
        })

      {init_fn, _predict_fn} = Axon.build(qlora_model, mode: :train, seed: 17, compiler: exla)
      init_state = init_fn.(Nx.template({2, 4}, :f32), Axon.ModelState.empty())

      trained_state =
        qlora_model
        |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.sgd(learning_rate: 0.01))
        |> Axon.Loop.run(training_batches(), init_state, epochs: 1, compiler: exla)

      assert match?(%Axon.ModelState{}, trained_state)
      assert Enum.all?(Map.keys(trained_state.parameters), &String.starts_with?(&1, "lora_"))
      refute Map.has_key?(trained_state.data, "query")
      refute Map.has_key?(trained_state.data, "key")
      refute Map.has_key?(trained_state.data, "value")
    else
      assert true
    end
  end

  defp qkv_dense_model do
    Axon.input("input", shape: {nil, 4})
    |> Axon.dense(6, activation: :relu, name: "query")
    |> Axon.dense(6, activation: :relu, name: "key")
    |> Axon.dense(3, name: "value")
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
end
