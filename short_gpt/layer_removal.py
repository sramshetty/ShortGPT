import torch

import transformers


def layer_removal(model, layers_to_remove={}):

    for layer_name, layer_idx in layers_to_remove.items():
        del getattr(model, layer_name)[layer_idx]
