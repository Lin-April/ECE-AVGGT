"""Inference-time AVGGT patch for VGGT.

This module does not change model weights. It patches the VGGT aggregator so
early global blocks run per frame and later global blocks attend to all queries
but only a grid-subsampled K/V set.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def apply_avggt(
    model,
    subsample_factor=4,
    tearly=9,
    ablate_idx=None,
    keep_first_frame=True,
    use_mean=True,
    preserve_diagonal=False,
    routing=None,
    per_layer_factor=None,
):
    """Patch a loaded VGGT model with AVGGT-style inference behavior.

    Args:
        model: A loaded ``vggt.models.vggt.VGGT`` instance.
        subsample_factor: K/V patch subsampling factor. Supported values are
            1, 2, 4, 6, and 9. ``1`` only applies early global-to-frame.
        tearly: Number of early global blocks to run as frame attention.
        keep_first_frame: Keep all patch K/V tokens for VGGT's reference frame.
        use_mean: Add one mean K/V token for dropped patch tokens.
        preserve_diagonal: Add explicit self-attention logits for every query.
            This is closer to the paper, but it materializes attention logits and
            can use much more memory than the default SDPA path.
        routing: Optional per-global-layer actions. Each action must be one of
            "skip", "frame", or "kv". When omitted, the original AVGGT
            tearly/subsample_factor behavior is preserved.
        per_layer_factor: Optional per-global-layer K/V subsampling factors for
            "kv" actions. When omitted, ``subsample_factor`` is used for every
            K/V layer.
    """
    if not hasattr(model, "aggregator"):
        raise TypeError("apply_avggt expects a VGGT model with an aggregator")

    aggregator = model.aggregator
    aggregator.avggt_tearly = int(tearly)
    aggregator.avggt_ablate_idx = ablate_idx
    aggregator.avggt_subsample_factor = int(subsample_factor)
    aggregator.avggt_keep_first_frame = bool(keep_first_frame)
    aggregator.avggt_use_mean = bool(use_mean)
    aggregator.avggt_preserve_diagonal = bool(preserve_diagonal)
    num_global_blocks = len(aggregator.global_blocks)
    aggregator.avggt_routing = _validate_routing(routing, num_global_blocks)
    aggregator.avggt_per_layer_factor = _validate_per_layer_factor(
        per_layer_factor,
        num_global_blocks,
        int(subsample_factor),
    )

    if not hasattr(aggregator, "_vggt_process_global_attention"):
        aggregator._vggt_process_global_attention = aggregator._process_global_attention

    aggregator._process_global_attention = MethodType(_process_global_attention, aggregator)
    return model


def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    """Replacement for VGGT Aggregator._process_global_attention."""
    intermediates = []

    for _ in range(self.aa_block_size):
        if self.training:
            raise RuntimeError("AVGGT monkey patch is intended for inference only")

        action = _layer_action(self, global_idx)
        if action == "skip":
            pass
        elif action == "frame":
            tokens = _run_global_block_as_frame(self, tokens, B, S, P, C, global_idx, pos)
        elif action == "kv":
            if tokens.shape != (B, S * P, C):
                tokens = tokens.view(B, S, P, C).view(B, S * P, C)

            global_pos = pos
            if global_pos is not None and global_pos.shape != (B, S * P, 2):
                global_pos = global_pos.view(B, S, P, 2).view(B, S * P, 2)

            block = self.global_blocks[global_idx]
            factor = self.avggt_per_layer_factor[global_idx]
            if factor <= 1:
                tokens = block(tokens, pos=global_pos)
            else:
                tokens = _run_global_block_with_sga(
                    block=block,
                    x=tokens,
                    pos=global_pos,
                    B=B,
                    S=S,
                    P=P,
                    patch_start_idx=self.patch_start_idx,
                    factor=factor,
                    keep_first_frame=self.avggt_keep_first_frame,
                    use_mean=self.avggt_use_mean,
                    preserve_diagonal=self.avggt_preserve_diagonal,
                )
        else:
            raise RuntimeError(f"Unsupported AVGGT routing action: {action}")

        global_idx += 1
        intermediates.append(tokens.view(B, S, P, C))

    return tokens, global_idx, intermediates


def _validate_routing(routing, num_layers):
    if routing is None:
        return None
    routing = list(routing)
    if len(routing) != num_layers:
        raise ValueError(f"routing must have {num_layers} entries, got {len(routing)}")
    supported = {"skip", "frame", "kv"}
    invalid = sorted({action for action in routing if action not in supported})
    if invalid:
        raise ValueError(f"routing actions must be one of {sorted(supported)}, got {invalid}")
    return routing


def _validate_per_layer_factor(per_layer_factor, num_layers, default_factor):
    if per_layer_factor is None:
        factors = [default_factor] * num_layers
    else:
        factors = list(per_layer_factor)
        if len(factors) != num_layers:
            raise ValueError(f"per_layer_factor must have {num_layers} entries, got {len(factors)}")
        factors = [default_factor if factor is None else int(factor) for factor in factors]

    for factor in factors:
        _factor_to_stride(factor)
    return factors


def _layer_action(aggregator, global_idx):
    if getattr(aggregator, "avggt_ablate_idx", None) is not None:
        if aggregator.avggt_ablate_idx == global_idx:
            return "frame"
        return "kv"
    if aggregator.avggt_routing is not None:
        return aggregator.avggt_routing[global_idx]
    if global_idx < aggregator.avggt_tearly:
        return "frame"
    return "kv"


def _run_global_block_as_frame(aggregator, tokens, B, S, P, C, global_idx, pos):
    if tokens.shape != (B * S, P, C):
        tokens = tokens.view(B, S, P, C).view(B * S, P, C)

    frame_pos = pos
    if frame_pos is not None and frame_pos.shape != (B * S, P, 2):
        frame_pos = frame_pos.view(B, S, P, 2).view(B * S, P, 2)

    return aggregator.global_blocks[global_idx](tokens, pos=frame_pos)


def _run_global_block_with_sga(
    block,
    x,
    pos,
    B,
    S,
    P,
    patch_start_idx,
    factor,
    keep_first_frame,
    use_mean,
    preserve_diagonal,
):
    residual = _subsampled_attention(
        attn=block.attn,
        x=block.norm1(x),
        pos=pos,
        B=B,
        S=S,
        P=P,
        patch_start_idx=patch_start_idx,
        factor=factor,
        keep_first_frame=keep_first_frame,
        use_mean=use_mean,
        preserve_diagonal=preserve_diagonal,
    )
    x = x + block.drop_path1(block.ls1(residual))
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x


def _subsampled_attention(
    attn,
    x,
    pos,
    B,
    S,
    P,
    patch_start_idx,
    factor,
    keep_first_frame,
    use_mean,
    preserve_diagonal,
):
    _, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)

    if attn.rope is not None:
        q = attn.rope(q, pos)
        k = attn.rope(k, pos)

    keep_idx = _build_keep_indices(
        pos=pos,
        S=S,
        P=P,
        patch_start_idx=patch_start_idx,
        factor=factor,
        keep_first_frame=keep_first_frame,
        device=x.device,
    )
    dropped_idx = _build_dropped_patch_indices(S, P, patch_start_idx, keep_idx, x.device)

    k_keep = k.index_select(dim=2, index=keep_idx)
    v_keep = v.index_select(dim=2, index=keep_idx)

    if not preserve_diagonal:
        if use_mean and dropped_idx.numel() > 0:
            k_mean = k.index_select(dim=2, index=dropped_idx).mean(dim=2, keepdim=True)
            v_mean = v.index_select(dim=2, index=dropped_idx).mean(dim=2, keepdim=True)
            k_keep = torch.cat([k_keep, k_mean], dim=2)
            v_keep = torch.cat([v_keep, v_mean], dim=2)

        output = F.scaled_dot_product_attention(
            q,
            k_keep,
            v_keep,
            dropout_p=attn.attn_drop.p if attn.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(B, N, C)
        output = attn.proj(output)
        return attn.proj_drop(output)

    q_scaled = q * attn.scale
    keep_logits = q_scaled @ k_keep.transpose(-2, -1)
    self_logits = (q_scaled * k).sum(dim=-1, keepdim=True)

    logits = [keep_logits, self_logits]
    value_parts = [v_keep, v]
    if use_mean and dropped_idx.numel() > 0:
        k_mean = k.index_select(dim=2, index=dropped_idx).mean(dim=2, keepdim=True)
        v_mean = v.index_select(dim=2, index=dropped_idx).mean(dim=2, keepdim=True)
        mean_logits = q_scaled @ k_mean.transpose(-2, -1)
        logits.append(mean_logits)
        value_parts.append(v_mean)

    weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)

    keep_count = k_keep.shape[2]
    output = weights[..., :keep_count] @ value_parts[0]
    offset = keep_count

    self_weight = weights[..., offset : offset + 1]
    output = output + self_weight * value_parts[1]
    offset += 1

    if len(value_parts) == 3:
        mean_weight = weights[..., offset : offset + 1]
        output = output + mean_weight * value_parts[2]

    output = output.transpose(1, 2).reshape(B, N, C)
    output = attn.proj(output)
    return attn.proj_drop(output)


def _build_keep_indices(pos, S, P, patch_start_idx, factor, keep_first_frame, device):
    sh, sw = _factor_to_stride(factor)
    patch_h, patch_w = _infer_patch_grid(pos, P, patch_start_idx)

    keep = []
    for frame_idx in range(S):
        frame_offset = frame_idx * P
        keep.extend(frame_offset + idx for idx in range(patch_start_idx))

        if keep_first_frame and frame_idx == 0:
            keep.extend(frame_offset + idx for idx in range(patch_start_idx, P))
            continue

        for row in range(0, patch_h, sh):
            for col in range(0, patch_w, sw):
                patch_idx = row * patch_w + col
                if patch_idx < P - patch_start_idx:
                    keep.append(frame_offset + patch_start_idx + patch_idx)

    return torch.tensor(sorted(set(keep)), device=device, dtype=torch.long)


def _build_dropped_patch_indices(S, P, patch_start_idx, keep_idx, device):
    keep = set(keep_idx.detach().cpu().tolist())
    dropped = []
    for frame_idx in range(S):
        frame_offset = frame_idx * P
        for idx in range(patch_start_idx, P):
            absolute_idx = frame_offset + idx
            if absolute_idx not in keep:
                dropped.append(absolute_idx)
    return torch.tensor(dropped, device=device, dtype=torch.long)


def _factor_to_stride(factor):
    strides = {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        6: (2, 3),
        9: (3, 3),
    }
    if factor not in strides:
        raise ValueError("AVGGT supports subsample_factor values: 1, 2, 4, 6, 9")
    return strides[factor]


def _infer_patch_grid(pos, P, patch_start_idx):
    patch_count = P - patch_start_idx
    side = int(patch_count**0.5)
    if side * side == patch_count:
        return side, side

    if pos is None:
        return 1, patch_count

    patch_pos = pos[0, patch_start_idx:P].detach().cpu()
    rows = torch.unique(patch_pos[:, 0]).numel()
    cols = torch.unique(patch_pos[:, 1]).numel()
    if rows * cols != patch_count:
        return 1, patch_count
    return int(rows), int(cols)
