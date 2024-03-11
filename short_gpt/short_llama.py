from typing import List, Optional, Tuple
from llama.model import ModelArgs
from llama.tokenizer import Tokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama import Transformer

from metrics import *


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class TransformerWrapper(Transformer):
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, return_hiddens: bool = False):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        hiddens = [h]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            if return_hiddens:
                hiddens.append(h)

        h = self.norm(h)
        output = self.output(h).float()

        if return_hiddens:
            return output, hiddens

        return output
        

class ShortLlama():

    def __init__(self, llama):
        self.llama = llama
        checkpoint = self.llama.model.state_dict()
        self.llama.model = TransformerWrapper(self.llama.model.params)
        self.llama.model.load_state_dict(checkpoint, strict=False)

        self.importances = [0 for _ in self.llama.model.layers]

    def remove_layers(
        self,
        layers_to_remove=[],
        num_layers=None,
    ):
        if not layers_to_remove and num_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:num_layers].tolist()

        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.llama.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove

    @torch.inference_mode()
    def eval_importance(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.llama.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.llama.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        if min_prompt_len == total_len:
            _, hiddens = self.llama.model.forward(tokens, prev_pos, return_hiddens=True)

            for i in range(len(hiddens)-1):
                h_pair = hiddens[i:i+2]
                self.importances[i] += block_influence(h_pair[0], h_pair[1]).sum().cpu().item()
        
        return
