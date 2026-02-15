defmodule Lorax do
  @moduledoc """
  Simple Low-Rank Adaptation (LoRA) implementation

  ## LoRA model creation
  To create a LoRA model, freeze an existing model and inject LoRA layers using `Lorax.inject/2`.

  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_key: false,
      target_query: false,
      target_value: true
    })
  ```

  For more detailed guides, see
  1. [Finetuning LLMs with LoRA](finetuning_gpt_with_lora.livemd)
  2. [Running LLMs with LoRA](running_gpt_with_lora.livemd)


  LoRA layers are implemented by injecting new nodes into the Axon struct.
  These LoRA nodes represent the B and A matrices. Each node takes an input `x` and computes `BAx`.
  Furthermore, the LoRA node will receive `Wx` as an input and compute `Wx + BAx`.
  This isn't the standard implementation, but it simplifies the injection process.

  ## Injection Process

  Beginning state
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1]
  </div>

  Create an empty dummy node
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1] --> C[dummy id:2]
  </div>

  Create lora node with input ids = [0, 2]
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1] --> C[dummy id:2] --> E[lora id:3]
    A[input id:0] --> E[lora id:3]
  </div>

  target takes dummy's id, throw away dummy node
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> C[target id:2]
    C[target id:2] --> E[lora id:3]
    A[input id:0] --> E[lora id:3]
  </div>


  lora takes target's original id
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> C[target id:2] --> E[lora id:1]
    A[input id:0] --> E[lora id:1]
  </div>


  lora and target are now swapped.
  Any downstream node that relied on node id:1 will now receive `Wx + BAx`
  """

  import Nx.Defn
  alias Lorax.QLoRA.Target, as: QLoRATarget

  defmodule Config do
    @moduledoc """
    Config for `Lorax.inject/2`

    `r` is the rank in the low-rank matrices used in LoRA.
    A higher value of r increases the expressiveness of the adaptation,
    However, it also increases the number of parameters and the computational
    cost. Conversely, a lower value of r makes the adaptation simpler and less
    resource-intensive. Defaults to 1.

    `alpha` is a scaling factor that controls the magnitude of changes introduced
    by the low-rank matrices. A higher value of `alpha` means that the
    modifications made by LoRA have a greater impact on the model's original
    weights. This can lead to more significant changes in the model's behavior.
    A lower value results in more subtle changes. Defaults to 2.

    `dropout` specifies the dropout rate applied to the low-rank matrices.

    `dropout_seed` determines the seed used for `Nx.Random.key/1` during
    dropout application. When defined, it ensures that the LoRA adapter
    produces consistent tensor values, assuming that other layers also have
    deterministic outputs.

    `param_type` specifies the numerical representation for the A and B
    matrices. Defaults to float32

    `target_query` specifies whether to apply LoRA to all query matrices in an
    attention block. Defaults to true.

    `target_value` specifies whether to apply LoRA to all value matrices in an
    attention block. Defaults to true.

    `target_key` specifies whether to apply LoRA to all key matrices in an
    attention block. Defaults to true.

    `qlora_targets` specifies a map keyed by target layer name, containing
    `%Lorax.QLoRA.Target{}` definitions for QLoRA execution.
    If provided, every selected target node must have an entry.
    """
    defstruct r: 1,
              alpha: 2,
              dropout: 0.0,
              dropout_seed: nil,
              param_type: {:f, 32},
              target_query: true,
              target_key: true,
              target_value: true,
              target_node_fn: nil,
              qlora_targets: %{}
  end

  @doc """
  Returns a modified Axon model with LoRA nodes inserted according to the provided configuration.

  `target_key`, `target_query`, `target_value` are required if `target_node_fn` isn't specified

  ## Examples
  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_key: true,
      target_query: true,
      target_value: true
    })
  ```

  ## Targeting nodes manually
  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_node_fn: fn %Axon.Node{name: name_fn} ->
        # names are generated lazily, and look like "decoder.blocks.11.self_attention.value"
        # have to invoke the function to see what layer the node represents
        # https://github.com/elixir-nx/axon/blob/v0.6.0/lib/axon.ex#L3923
        name = name_fn.(nil, nil)
        shortname = String.split(name, ".") |> List.last()

        if shortname == "output" do
          true
        else
          false
        end
      end
    })
  ```
  """
  def inject(%Axon{} = axon, %Config{} = config) do
    target_nodes = get_target_nodes(axon, config)
    qlora_targets = Lorax.QLoRA.validate_targets!(axon, target_nodes, config.qlora_targets)

    Enum.reduce(target_nodes, axon, fn target_id, %Axon{nodes: acc_nodes} = acc ->
      # Grab our target node, create a fake Axon container for it
      %Axon.Node{} = target_node = Map.fetch!(acc_nodes, target_id)
      target_name = node_name(target_node)
      target_axon = %Axon{acc | output: target_id}

      # Get its parent and create fake Axon containers
      # Note: The parent field of Axon.Node is usually a list,
      #       but for our purposes, it's just a list of one input
      parent_ids = target_node.parent
      parent_axons = Enum.map(parent_ids, fn id -> %Axon{acc | output: id} end)

      # Create a dummy Axon container for target to move into
      dummy_axon = %Axon{output: dummy_id} = Axon.nx(target_axon, fn x -> x end)

      # lora node takes target's place
      # target node takes dummy's place
      injected_node =
        case Map.get(qlora_targets, target_name) do
          nil ->
            {create_lora_node(target_node, parent_axons, dummy_axon, config), %{}}

          %QLoRATarget{} = qlora_target ->
            %Axon{output: injected_id, nodes: injected_nodes} =
              create_qlora_node(target_node, parent_axons, dummy_axon, config, qlora_target)

            %Axon.Node{} = injected_node = injected_nodes[injected_id]

            extra_nodes =
              injected_nodes
              |> Map.drop(Map.keys(acc_nodes))
              |> Map.delete(injected_id)

            {injected_node, extra_nodes}
        end

      {injected_node, extra_nodes} = injected_node
      injected_node = %Axon.Node{injected_node | id: target_id}
      target_node = %Axon.Node{target_node | id: dummy_id}

      # update Axon container's map of nodes so that
      # 1. whenever downstream nodes reference target_id, it'll now point to our lora node
      # 2. whenever lora node references dummy id, it'll take the output value (Wx) from target
      new_nodes =
        acc_nodes
        |> Map.merge(extra_nodes)
        |> Map.put(target_id, injected_node)
        |> Map.put(dummy_id, target_node)

      %Axon{acc | nodes: new_nodes}
    end)
  end

  defp create_lora_node(
         %Axon.Node{op: :conv, name: target_name_fn, opts: opts, parameters: parameters},
         parent_axons,
         dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    {a_shape, b_shape} = Lorax.Shape.calc_ab(:conv, r, parameters)

    lora_A = Axon.param("lora_down", a_shape, initializer: :normal, type: param_type)
    lora_B = Axon.param("lora_up", b_shape, initializer: :zeros, type: param_type)
    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&lora_conv_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      layer_opts: opts,
      scaling: scaling
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  # Parent + dummy axon are inputs to create the lora node
  # target_node_name_fn is provided to help create a name for our new lora node
  defp create_lora_node(
         %Axon.Node{op: :dense, name: target_name_fn, parameters: parameters},
         parent_axons,
         dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    {a_shape, b_shape} = Lorax.Shape.calc_ab(:dense, r, parameters)

    lora_A = Axon.param("lora_down", a_shape, initializer: :normal, type: param_type)
    lora_B = Axon.param("lora_up", b_shape, initializer: :zeros, type: param_type)
    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&lora_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      scaling: scaling
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  defp create_qlora_node(
         %Axon.Node{op: :conv, name: target_name_fn, opts: opts},
         parent_axons,
         _dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         },
         %QLoRATarget{
           packed_weight: packed_weight,
           provider: provider,
           provider_opts: provider_opts,
           base_bias: base_bias,
           base_type: base_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    {a_shape, b_shape} = calc_qlora_ab_from_packed!(:conv, r, packed_weight)

    packed_weight = create_fixed_state_param("packed_weight", packed_weight)
    base_bias = create_base_bias_param(base_bias, base_type)
    lora_A = Axon.param("lora_down", a_shape, initializer: :normal, type: param_type)
    lora_B = Axon.param("lora_up", b_shape, initializer: :zeros, type: param_type)
    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&qlora_conv_impl/6, parent_axons ++ [packed_weight, base_bias, lora_A, lora_B],
      op_name: :qlora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      provider: provider,
      provider_opts: provider_opts,
      base_type: base_type,
      layer_opts: opts,
      scaling: scaling
    )
  end

  defp create_qlora_node(
         %Axon.Node{op: :dense, name: target_name_fn},
         parent_axons,
         _dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         },
         %QLoRATarget{
           packed_weight: packed_weight,
           provider: provider,
           provider_opts: provider_opts,
           base_bias: base_bias,
           base_type: base_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    {a_shape, b_shape} = calc_qlora_ab_from_packed!(:dense, r, packed_weight)

    packed_weight = create_fixed_state_param("packed_weight", packed_weight)
    base_bias = create_base_bias_param(base_bias, base_type)
    lora_A = Axon.param("lora_down", a_shape, initializer: :normal, type: param_type)
    lora_B = Axon.param("lora_up", b_shape, initializer: :zeros, type: param_type)
    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&qlora_dense_impl/6, parent_axons ++ [packed_weight, base_bias, lora_A, lora_B],
      op_name: :qlora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      provider: provider,
      provider_opts: provider_opts,
      base_type: base_type,
      scaling: scaling
    )
  end

  defnp lora_conv_impl(x, wx, lora_A, lora_B, opts \\ []) do
    scaling = opts[:scaling]
    layer_opts = opts[:layer_opts]

    after_a = Axon.Layers.conv(x, lora_A, layer_opts)
    after_b = Axon.Layers.conv(after_a, lora_B)
    bax = Nx.multiply(after_b, scaling)
    Nx.add(wx, bax)
  end

  defnp lora_impl(x, wx, lora_A, lora_B, opts \\ []) do
    dropout = opts[:dropout]
    dropout_seed = opts[:dropout_seed]
    scaling = opts[:scaling]

    x = Axon.Layers.dropout(x, Nx.Random.key(dropout_seed), rate: dropout)
    after_a = Axon.Layers.dense(x, lora_A |> Nx.transpose())
    after_b = Nx.dot(after_a, lora_B |> Nx.transpose())
    bax = Nx.multiply(after_b, scaling)

    Nx.add(wx, bax)
  end

  defnp qlora_dense_impl(x, packed_weight, base_bias, lora_A, lora_B, opts \\ []) do
    dropout = opts[:dropout]
    dropout_seed = opts[:dropout_seed]
    scaling = opts[:scaling]
    provider = opts[:provider]
    provider_opts = opts[:provider_opts]
    base_type = opts[:base_type]

    dequantized_weight =
      provider.dequant(packed_weight, provider_opts)
      |> Nx.as_type(base_type)
      |> Nx.Defn.Kernel.stop_grad()

    frozen_base_bias =
      base_bias
      |> Nx.as_type(base_type)
      |> Nx.Defn.Kernel.stop_grad()

    wx = Axon.Layers.dense(x, dequantized_weight, frozen_base_bias)

    x = Axon.Layers.dropout(x, Nx.Random.key(dropout_seed), rate: dropout)
    after_a = Axon.Layers.dense(x, lora_A |> Nx.transpose())
    after_b = Nx.dot(after_a, lora_B |> Nx.transpose())
    bax = Nx.multiply(after_b, scaling)

    Nx.add(wx, bax)
  end

  defnp qlora_conv_impl(x, packed_weight, base_bias, lora_A, lora_B, opts \\ []) do
    scaling = opts[:scaling]
    layer_opts = opts[:layer_opts]
    provider = opts[:provider]
    provider_opts = opts[:provider_opts]
    base_type = opts[:base_type]

    dequantized_weight =
      provider.dequant(packed_weight, provider_opts)
      |> Nx.as_type(base_type)
      |> Nx.Defn.Kernel.stop_grad()

    frozen_base_bias =
      base_bias
      |> Nx.as_type(base_type)
      |> Nx.Defn.Kernel.stop_grad()

    wx = Axon.Layers.conv(x, dequantized_weight, frozen_base_bias, layer_opts)
    after_a = Axon.Layers.conv(x, lora_A, layer_opts)
    after_b = Axon.Layers.conv(after_a, lora_B)
    bax = Nx.multiply(after_b, scaling)

    Nx.add(wx, bax)
  end

  defp create_base_bias_param(nil, base_type) do
    create_fixed_state_param("base_bias", Nx.tensor(0, type: base_type))
  end

  defp create_base_bias_param(%Nx.Tensor{} = base_bias, _base_type) do
    create_fixed_state_param("base_bias", base_bias)
  end

  defp create_fixed_state_param(name, %Nx.Tensor{} = tensor) do
    Axon.param(name, Nx.shape(tensor),
      type: Nx.type(tensor),
      kind: :state,
      initializer: fn _shape, _type, _key -> tensor end
    )
  end

  defp calc_qlora_ab_from_packed!(:dense, r, %Nx.Tensor{} = packed_weight) do
    case Nx.shape(packed_weight) do
      {in_features, out_features} ->
        {{r, in_features}, {out_features, r}}

      shape ->
        raise ArgumentError,
              "QLoRA dense packed weight must be rank-2, got shape #{inspect(shape)}"
    end
  end

  defp calc_qlora_ab_from_packed!(:conv, r, %Nx.Tensor{} = packed_weight) do
    case Nx.shape(packed_weight) do
      {kernel_h, kernel_w, in_features, out_features} ->
        {{kernel_h, kernel_w, in_features, r}, {1, 1, r, out_features}}

      shape ->
        raise ArgumentError,
              "QLoRA conv packed weight must be rank-4, got shape #{inspect(shape)}"
    end
  end

  defp create_name_fn(target_name_fn) do
    fn op, op_count ->
      target_name = target_name_fn.(op, op_count)

      "lora_" <> target_name
    end
  end

  defp node_name(%Axon.Node{name: name_fn}) do
    name_fn.(nil, nil)
  end

  defp get_target_nodes(axon, %Config{target_node_fn: target_node_fn})
       when is_function(target_node_fn, 1) do
    Axon.reduce_nodes(axon, [], fn %Axon.Node{id: id} = node, acc ->
      if target_node_fn.(node) do
        [id | acc]
      else
        acc
      end
    end)
  end

  # note: This is just for LLMs
  defp get_target_nodes(
         axon,
         %Config{
           target_query: target_query,
           target_key: target_key,
           target_value: target_value
         }
       ) do
    Axon.reduce_nodes(axon, [], fn
      %Axon.Node{id: id, name: name_fn, op: :dense}, acc ->
        shortname =
          name_fn.(:dense, nil)
          |> String.split(".")
          |> List.last()

        if (target_key and shortname == "key") or
             (target_query and shortname == "query") or
             (target_value and shortname == "value") do
          [id | acc]
        else
          acc
        end

      %Axon.Node{}, acc ->
        acc
    end)
  end
end
