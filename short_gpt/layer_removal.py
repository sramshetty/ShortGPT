from collections import OrderedDict

import torch.nn as nn


def layer_removal(
    model: nn.Module,
    layers_to_remove: OrderedDict
):
    """
    Generic removal implementation
    """

    for layer_name, layer_idx in layers_to_remove.items():
        modules = layer_name.split(".")
        mod = model
        for m in modules[:-1]:
            mod = getattr(mod, m)
        
        if layer_idx is None:
            del getattr(mod, modules[-1])
        else:
            del getattr(mod, modules[-1])[layer_idx]
