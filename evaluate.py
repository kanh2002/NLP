#!/usr/bin/env python3
import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    uas = 0
    las = 0

    for words, pos, heads, rels, mask in loader:
        pos   = pos.to(device)
        heads = heads.to(device)
        rels  = rels.to(device)
        mask  = mask.to(device)

        arc_scores, rel_scores = model(words, pos)

        # predict heads
        pred_heads = arc_scores.argmax(dim=-1)

        # predict rels at predicted heads
        B, L, _ = arc_scores.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        i_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pred_rels = rel_scores[b_idx, i_idx, pred_heads].argmax(dim=-1)

        # valid tokens: real words only
        valid = mask & (heads >= 0)

        total += valid.sum().item()
        uas += (pred_heads[valid] == heads[valid]).sum().item()
        las += ((pred_heads[valid] == heads[valid]) &
                (pred_rels[valid] == rels[valid])).sum().item()

    return uas / total, las / total
