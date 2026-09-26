"""Preserve the 1.5B ToolRL objective while splitting execution batches.

The original runtime sums each logical micro-batch's masked token mean,
divided by the configured (not dynamic) accumulation count. Splitting a
logical batch must NOT give its smaller pieces equal weight.
"""

import torch


def split_logical_batches(logical_batches, execution_token_limit, accumulation):
    from verl.utils.seqlen_balancing import rearrange_micro_batches

    if accumulation < 1 or execution_token_limit < 1:
        raise ValueError('Positive execution budget and accumulation required')
    parts, normalizers = [], []
    for logical in logical_batches:
        response_length = logical['responses'].shape[-1]
        count = int(logical['attention_mask'][:, -response_length:].sum())
        if count <= 0:
            raise ValueError('Logical batch has no response tokens')
        smaller, _ = rearrange_micro_batches(logical, execution_token_limit)
        parts.extend(smaller)
        normalizers.extend([1.0 / (count * accumulation)] * len(smaller))
    return parts, normalizers


def policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange,
                normalization_factor, **kwargs):
    """Original unclamped log-ratio PPO loss, with explicit token weights."""
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    first = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
    loss = (torch.maximum(first, clipped) * eos_mask).sum() * normalization_factor
    count = eos_mask.sum().clamp_min(1)
    clip_fraction = ((clipped > first).float() * eos_mask).sum() / count
    kl = (-log_ratio * eos_mask).sum() / count
    return loss, clip_fraction, kl
