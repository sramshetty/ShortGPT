from typing import List

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import *


class ShortHFModel():

    def __init__(self, model_name: str, layers_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        # self.model.params = self.model.to_fp16(self.model.params)
        self.model.to("cuda")

        modules = layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod
        self.importances = [0 for _ in self.layers]  # layer-wise importance scores

    def remove_layers(
        self,
        layers_to_remove: List[int] = [],
        num_layers: int = 0,
    ):
        if not layers_to_remove and num_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:num_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor]):
        for i in range(len(hiddens)-1):
            h_pair = hiddens[i:i+2]
            self.importances[i] += block_influence(h_pair[0], h_pair[1]).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9
    ):
        """
        Computes layer-wise importances over input texts.

        NOTE: Paper performs no generation during importance computation, which suggests a `max_gen_len`= 0.

        Args:
            prompts (List[str]): List of prompts.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.

        Returns:
            None
        """
        
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        # authors use a sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids  # ensure 2d
            inputs = input_ids[seq_ids, start:start+max_seq_len]
            attn = attn_mask[seq_ids, start:start+max_seq_len]

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            
            self.compute_bi(outputs.hidden_states)

        return

        