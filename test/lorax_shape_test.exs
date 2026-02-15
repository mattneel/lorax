defmodule Lorax.ShapeTest do
  use ExUnit.Case, async: true

  test "calc_ab/3 for dense infers adapter dimensions from x and wx shapes" do
    {a_shape_fn, b_shape_fn} = Lorax.Shape.calc_ab(:dense, 2, [])

    assert a_shape_fn.({8, 16}, {8, 32}) == {2, 16}
    assert b_shape_fn.({8, 16}, {8, 32}) == {32, 2}
  end

  test "calc_ab/3 for conv uses kernel size and channels from parameters" do
    conv_model =
      Axon.input("image", shape: {nil, 8, 8, 3})
      |> Axon.conv(4, kernel_size: {3, 3}, name: "conv")

    {_id, %Axon.Node{parameters: parameters}} =
      Enum.find(conv_model.nodes, fn {_id, %Axon.Node{op: op}} -> op == :conv end)

    {a_shape_fn, b_shape_fn} = Lorax.Shape.calc_ab(:conv, 2, parameters)

    assert a_shape_fn.({2, 8, 8, 3}, {2, 6, 6, 4}) == {3, 3, 3, 2}
    assert b_shape_fn.({2, 8, 8, 3}, {2, 6, 6, 4}) == {1, 1, 2, 4}
  end
end
