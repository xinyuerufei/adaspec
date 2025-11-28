from contextlib import nullcontext

import torch
from tqdm import tqdm

from sampling.utils import norm_logits, sample, max_fn


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Decoder(metaclass=Singleton):
    def __init__(self):
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, s: str, return_tensors='pt') -> torch.Tensor:
        return self.tokenizer.encode(s, return_tensors=return_tensors)

    def decode(self, t: torch.Tensor) -> list:
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in t]


import torch.nn as nn
import torch.optim as optim


@torch.no_grad()
def speculative_sampling(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                         max_len: int, gamma: int = 4,
                         temperature: float = 1, top_k: int = 0, top_p: float = 0,
                         random_seed: int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    accepted_count = 0
    rejected_count = 0

    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :],
                                              temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)

            # normalize the logits
            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :],
                                         temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :],
                                         temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)

            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device=p.device)
                j = x[:, prefix_len + i]

                if r < torch.min(torch.tensor([1], device=q.device),
                                 p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                    accepted_count += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    rejected_count += 1
                    break

            prefix = x[:, :n + 1]

            if is_all_accept:
                t = sample(p[:, -1, :])

            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix, accepted_count, rejected_count


@torch.no_grad()
def my_speculative_sampling(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module,
                            max_len: int, eos, gamma: int = 4,
                            temperature: float = 1, top_k: int = 0, top_p: float = 0,
                            random_seed: int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization

    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    accepted_count = 0
    rejected_count = 0

    assert prefix.shape[0] == 1, "input batch size must be 1"

    with nullcontext():
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = q[:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat((x, next_tok), dim=1)

            p = target_model(x).logits

            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if p[:, prefix_len + i - 1, :].argmax(dim=-1) == x[:, prefix_len + i]:
                    # accept, and update n
                    n += 1
                    accepted_count += 1
                else:
                    # reject
                    t = p[:, n, :].argmax(dim=-1, keepdim=True)
                    is_all_accept = False
                    rejected_count += 1
                    break

            prefix = x[:, :n + 1]

            if is_all_accept:
                t = p[:, -1, :].argmax(dim=-1, keepdim=True)

            prefix = torch.cat((prefix, t), dim=1)
            if prefix.eq(eos).any():
                idx = (prefix[0] == eos).nonzero()[0]
                prefix = prefix[0, :idx + 1].unsqueeze(0)
                break

    return prefix, accepted_count, rejected_count
