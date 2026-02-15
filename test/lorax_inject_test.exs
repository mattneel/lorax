defmodule Lorax.InjectTest do
  use ExUnit.Case, async: true

  test "inject/2 adds LoRA nodes for query/key/value dense layers and preserves output shape" do
    model = qkv_dense_model()
    lora_model = Lorax.inject(model, %Lorax.Config{})

    lora_layer_names =
      lora_model
      |> model_node_names()
      |> Enum.filter(&String.starts_with?(&1, "lora_"))
      |> Enum.sort()

    assert lora_layer_names == ["lora_key", "lora_query", "lora_value"]

    {base_init, base_predict} = Axon.build(model, seed: 7)
    {lora_init, lora_predict} = Axon.build(lora_model, seed: 7)

    input = Nx.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    template = Nx.template({2, 4}, :f32)

    base_state = base_init.(template, Axon.ModelState.empty())
    lora_state = lora_init.(template, Axon.ModelState.empty())

    assert Nx.shape(base_predict.(base_state, input)) ==
             Nx.shape(lora_predict.(lora_state, input))
  end

  test "inject/2 supports conv targets through target_node_fn and preserves output shape" do
    model =
      Axon.input("image", shape: {nil, 8, 8, 3})
      |> Axon.conv(4, kernel_size: {3, 3}, name: "conv")

    config = %Lorax.Config{
      r: 2,
      alpha: 4,
      target_node_fn: fn %Axon.Node{op: op, name: name_fn} ->
        op == :conv and name_fn.(nil, nil) == "conv"
      end
    }

    lora_model = Lorax.inject(model, config)

    lora_layer_names =
      lora_model
      |> model_node_names()
      |> Enum.filter(&String.starts_with?(&1, "lora_"))

    assert lora_layer_names == ["lora_conv"]

    {base_init, base_predict} = Axon.build(model, seed: 7)
    {lora_init, lora_predict} = Axon.build(lora_model, seed: 7)

    input = Nx.iota({2, 8, 8, 3}, type: :f32)
    template = Nx.template({2, 8, 8, 3}, :f32)

    base_state = base_init.(template, Axon.ModelState.empty())
    lora_state = lora_init.(template, Axon.ModelState.empty())

    assert Nx.shape(base_predict.(base_state, input)) ==
             Nx.shape(lora_predict.(lora_state, input))

    assert Nx.shape(lora_state.data["lora_conv"]["lora_down"]) == {3, 3, 3, 2}
    assert Nx.shape(lora_state.data["lora_conv"]["lora_up"]) == {1, 1, 2, 4}
  end

  defp qkv_dense_model do
    Axon.input("input", shape: {nil, 4})
    |> Axon.dense(6, activation: :relu, name: "query")
    |> Axon.dense(6, activation: :relu, name: "key")
    |> Axon.dense(3, name: "value")
  end

  defp model_node_names(%Axon{} = model) do
    Enum.map(model.nodes, fn {_id, %Axon.Node{name: name_fn}} -> name_fn.(nil, nil) end)
  end
end
