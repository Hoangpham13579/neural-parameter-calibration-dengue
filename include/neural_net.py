from typing import Any, List, Sequence, Union

import torch
from torch import nn

from .utils import random_tensor

# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility functions ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def sigmoid(beta=torch.tensor(1.0)):
    """Extends the torch.nn.sigmoid activation function by allowing for a slope parameter."""

    return lambda x: torch.sigmoid(beta * x)


# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).


ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
    "softmax": [lambda: torch.nn.Softmax(dim=-1), True],
}


def get_architecture(
    input_size: int, output_size: int, n_layers: int, cfg: dict
) -> List[int]:
    # Apply default to all hidden layers
    _nodes = [cfg.get("default")] * n_layers

    # Update layer-specific settings
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_size in _layer_specific.items():
        _nodes[layer_id] = layer_size

    return [input_size] + _nodes + [output_size]


def get_activation_funcs(n_layers: int, cfg: dict) -> List[callable]:
    """Extracts the activation functions from the config. The config is a dictionary containing the
    default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as linear layers.

    .. Example:
        activation_funcs:
          default: relu
          layer_specific:
            0: ~
            2: tanh
            3:
              name: HardTanh
              args:
                - -2  # min_value
                - +2  # max_value
    """

    def _single_layer_func(layer_cfg: Union[str, dict]) -> callable:
        """Return the activation function from an entry for a single layer"""

        # Entry is a single string
        if isinstance(layer_cfg, str):
            _f = ACTIVATION_FUNCS[layer_cfg.lower()]
            if _f[1]:
                return _f[0]()
            else:
                return _f[0]

        # Entry is a dictionary containing args and kwargs
        elif isinstance(layer_cfg, dict):
            _f = ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
            if _f[1]:
                return _f[0](*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
            else:
                return _f[0]

        elif layer_cfg is None:
            _f = ACTIVATION_FUNCS["linear"][0]

        else:
            raise ValueError(f"Unrecognized activation function {cfg}!")

    # Use default activation function on all layers
    _funcs = [_single_layer_func(cfg.get("default"))] * (n_layers + 1)

    # Change activation functions on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _layer_specific.items():
        _funcs[layer_id] = _single_layer_func(layer_cfg)

    return _funcs


def get_dropout_layers(n_layers: int, cfg: dict) -> List[callable]:
    """Extracts dropout layers from the config. The config is a dictionary containing the
    default dropout probability (or configuration) and a layer-specific entry detailing exceptions
    from the default. 'None' entries are interpreted as no dropout (i.e. identity).

    .. Example:
        dropout:
          default: 0.5
          layer_specific:
            0: 0.0
            2: 0.3
            3: 0  # Equivalently, no dropout on this layer
    """

    def _single_dropout_layer(layer_cfg: Union[float, dict, None]):
        """Return a dropout layer from a single configuration entry for a layer"""
        # If configuration is None or zero, return an identity layer (no dropout)
        if layer_cfg is None or (
            isinstance(layer_cfg, (float, int)) and float(layer_cfg) == 0
        ):
            return nn.Identity()

        # If the config is a float/int, assume it's the dropout probability
        if isinstance(layer_cfg, (float, int)):
            return nn.Dropout(p=float(layer_cfg))
        else:
            raise ValueError(f"Unrecognized dropout configuration: {layer_cfg}!")

    # Default dropout layer applied to all layers
    _layers = [_single_dropout_layer(cfg.get("default"))] * (n_layers + 1)

    # Update with layer-specific dropout configurations if provided
    _specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _specific.items():
        _layers[layer_id] = _single_dropout_layer(layer_cfg)

    return _layers


def get_bias(n_layers: int, cfg: dict) -> List[Any]:
    """Extracts the bias initialisation settings from the config. The config is a dictionary containing the
    default, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as unbiased layers.

    .. Example:
        biases:
          default: ~
          layer_specific:
            0: [-1, 1]
            3: [2, 3]
    """

    # Use the default value on all layers
    biases = [cfg.get("default")] * (n_layers + 1)

    # Amend bias on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_bias in _layer_specific.items():
        biases[layer_id] = layer_bias

    return biases


def get_layernorm_layers(
    n_layers: int, cfg: dict, architecture: List[int]
) -> List[callable]:
    """Extracts layer normalisation layers from the config. The config is a dictionary containing the
    default True/False (notify applying it or not) and a layer-specific entry detail application
    from the default. 'None' entries are interpreted as no batch normalisation (i.e. identity).

    .. Example:
        layernorm:
          default: True
          layer_specific:
            3: False  # Equivalently, no batchnorm on this layer
    """
    layers = list()
    for i in range(n_layers + 1):
        layer_cfg = cfg["layer_specific"].get(i, cfg["default"])
        output_dim = architecture[i + 1]

        if layer_cfg:
            layers.append(nn.LayerNorm(output_dim))
        else:
            layers.append(nn.Identity())
    return layers


# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):
    OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        dropout: dict,
        layernorm: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 500,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """
        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param dropout: a dictionary specifying the dropout probability for each layer
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to Itot the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        """

        super().__init__()
        self.flatten = nn.Flatten(start_dim=0)

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(
            input_size, output_size, num_layers, nodes_per_layer
        )
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        self.dropout_layers = get_dropout_layers(num_layers, dropout)
        self.layernorm_layers = get_layernorm_layers(
            num_layers, layernorm, self.architecture
        )
        self.bias = get_bias(num_layers, biases)

        # Add the neural net layers
        self.linear_layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias=self.bias[i] is not None,
            )
            # Weight init to solve gradient vanishing by ensuring variance of gradients, activation remain the same across layers.
            self.weight_initializer(layer, self.activation_funcs[i])

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] is not None:
                # Use the pytorch default if indicated
                if self.bias[i] == "default":
                    torch.nn.init.uniform_(layer.bias)
                # Initialise the bias on explicitly provided intervals
                else:
                    torch.nn.init.uniform_(layer.bias, self.bias[i][0], self.bias[i][1])

            self.linear_layers.append(layer)

        # Get the optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](
            self.parameters(), lr=learning_rate, **optimizer_kwargs
        )
        self.learning_rate = learning_rate

        # Get the initial distribution and initialise
        self.prior_distribution = prior
        self.initialise_to_prior(tol=prior_tol, max_iter=prior_max_iter)

    def weight_initializer(self, layer, activation_func):
        """
        Initialize the weights of a layer based on the provided activation function.
        """
        # Map activation functions to their suggested initializations. This dictionary can be extended or overridden by user config.
        init_map = {
            torch.nn.ReLU: lambda l: torch.nn.init.kaiming_uniform_(
                l.weight, mode="fan_in", nonlinearity="relu"
            ),
            torch.nn.LeakyReLU: lambda l: torch.nn.init.kaiming_uniform_(
                l.weight, mode="fan_in", nonlinearity="leaky_relu"
            ),
            torch.nn.ELU: lambda l: torch.nn.init.kaiming_uniform_(
                l.weight, mode="fan_in", nonlinearity="linear"
            ),
            torch.nn.GELU: lambda l: torch.nn.init.kaiming_uniform_(
                l.weight, mode="fan_in", nonlinearity="linear"
            ),
            # Default for Sigmoid, Tanh, Softmax and others
            "default": lambda l: torch.nn.init.xavier_uniform_(l.weight),
        }
        # Try to find the correct initializer based on the type of the activation function.
        for act_type, init_func in init_map.items():
            if activation_func == act_type:
                init_func(layer)
                break
            else:  # If none match, use a default initializer
                init_map["default"](layer)

    def initialise_to_prior(self, *, tol: float = 1e-5, max_iter: int = 500) -> None:
        """Initialises the neural net to output values following a prior distribution. The random tensor is drawn
        following a prior distribution and the neural network trained to output that value. Training is performed
        until the neural network output matches the drawn value (which typically only takes a few seconds), or until
        a maximum iteration count is reached.

        :param tol: the target error on the neural net initial output and drawn value.
        :param max_iter: maximum number of training steps to perform in the while loop
        """

        # If not initial distribution is given, nothing happens
        if self.prior_distribution is None:
            return

        # Draw a target tensor following the given prior distribution
        target = random_tensor(self.prior_distribution, size=(self.output_dim,))

        # Generate a prediction and train the net to output the given target
        prediction = self.forward(torch.rand(self.input_dim))
        iter = 0

        # Use a separate optimizer for the training
        optim = torch.optim.Adam(self.parameters(), lr=0.002)
        while torch.norm(prediction - target) > tol and iter < max_iter:
            prediction = self.forward(torch.rand(self.input_dim))
            loss = torch.nn.functional.mse_loss(target, prediction, reduction="sum")
            loss.backward()
            optim.step()
            optim.zero_grad()
            iter += 1

    # .................. Evaluation functions ..................

    # The model forward pass
    def forward(self, x):
        # Flatten before training
        x = self.flatten(x)
        for i, (linear, ln, activation, dropout) in enumerate(
            zip(
                self.linear_layers,
                self.layernorm_layers,
                self.activation_funcs,
                self.dropout_layers,
            )
        ):
            x = linear(x)
            x = ln(x)
            if activation is not None:
                x = activation(x)
            x = dropout(x)

        return x
