defmodule LoraxTest do
  use ExUnit.Case
  doctest Lorax

  test "config defaults are stable" do
    config = %Lorax.Config{}

    assert config.r == 1
    assert config.alpha == 2
    assert config.dropout == 0.0
    assert config.param_type == {:f, 32}
    assert config.target_query
    assert config.target_key
    assert config.target_value
  end
end
