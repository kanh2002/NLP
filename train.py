#!/usr/bin/env python3
import os
import time
import random
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from model import PhoBERTParser


# ======================= DATA =======================

def collate_fn(batch):
    words, pos, heads, rels = zip(*batch)
    B = len(words)
    L = max(len(w) for w in words)

    pos_ids = torch.zeros(B, L, dtype=torch.long)
    head_ids = torch.full((B, L), -1, dtype=torch.long)
    rel_ids = torch.full((B, L), -1, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)

    for i in range(B):
        l = len(words[i])
        pos_ids[i, :l] = pos[i]
        head_ids[i, :l] = heads[i]
        rel_ids[i, :l] = rels[i]
        mask[i, :l] = 1

    return list(words), pos_ids, head_ids, rel_ids, mask


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================= OPTIMIZER =======================

def make_optimizer(model, lr_phobert, lr_other, weight_decay):
    phobert_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "phobert" in name:
            phobert_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": phobert_params, "lr": lr_phobert},
            {"params": other_params, "lr": lr_other},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


# ======================= TRAIN =======================

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from dataset import read_conllu, DependencyDataset
    from evaluate import evaluate

    # -------- Load data --------
    train_sents = read_conllu(os.path.join(args.data_dir, "vi_vtb-ud-train.conllu"))
    dev_sents = read_conllu(os.path.join(args.data_dir, "vi_vtb-ud-dev.conllu"))
    test_sents = read_conllu(os.path.join(args.data_dir, "vi_vtb-ud-test.conllu"))

    pos2id = {"<pad>": 0}
    rel2id = {"<unk>": 0}
    for _, pos, _, rel in train_sents:
        for p in pos:
            pos2id.setdefault(p, len(pos2id))
        for r in rel:
            rel2id.setdefault(r, len(rel2id))

    train_ds = DependencyDataset(train_sents, pos2id, rel2id)
    dev_ds = DependencyDataset(dev_sents, pos2id, rel2id)
    test_ds = DependencyDataset(test_sents, pos2id, rel2id)

    train_ld = DataLoader(train_ds, args.batch_size, True, collate_fn=collate_fn)
    dev_ld = DataLoader(dev_ds, 1, False, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, 1, False, collate_fn=collate_fn)

    # -------- TensorBoard --------
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # -------- Model --------
    model = PhoBERTParser(
        len(pos2id),
        len(rel2id),
        hidden=args.hidden,
        mlp_size=args.mlp_size,
    ).to(device)

    optimizer = make_optimizer(
        model,
        args.lr_phobert,
        args.lr_other,
        args.weight_decay,
    )

    total_steps = len(train_ld) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    arc_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)
    rel_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    best_las = 0.0
    global_step = 0

    # ======================= LOOP =======================

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Progressive unfreeze
        model.freeze_phobert(epoch <= args.freeze_epochs)

        model.train()
        total_arc, total_rel, total_loss = 0, 0, 0

        for step, (words, pos, heads, rels, mask) in enumerate(train_ld):
            step_start = time.time()
            pos, heads, rels, mask = (
                pos.to(device),
                heads.to(device),
                rels.to(device),
                mask.to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.fp16):
                arc_logits, rel_logits_all = model(words, pos)
                B, L, _ = arc_logits.shape

                arc_loss = arc_loss_fn(arc_logits.view(B * L, L), heads.view(B * L))

                valid = mask & (heads >= 0)
                b = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
                i = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

                rel_logits = rel_logits_all[b[valid], i[valid], heads[valid]]
                rel_loss = rel_loss_fn(rel_logits, rels[valid])

                loss = args.arc_weight * arc_loss + args.rel_weight * rel_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # -------- Logging (STEP) --------
            writer.add_scalar("train/loss_arc", arc_loss.item(), global_step)
            writer.add_scalar("train/loss_rel", rel_loss.item(), global_step)
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/lr_phobert", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/lr_other", optimizer.param_groups[1]["lr"], global_step)
            writer.add_scalar("time/step_seconds", time.time() - step_start, global_step)

            total_arc += arc_loss.item()
            total_rel += rel_loss.item()
            total_loss += loss.item()
            global_step += 1

        # -------- Eval --------
        model.eval()
        with torch.no_grad():
            dev_uas, dev_las = evaluate(model, dev_ld, device)

        writer.add_scalar("dev/UAS", dev_uas, epoch)
        writer.add_scalar("dev/LAS", dev_las, epoch)
        writer.add_scalar("time/epoch_seconds", time.time() - epoch_start, epoch)

        print(
            f"[Epoch {epoch}] "
            f"Loss={total_loss:.3f} | "
            f"UAS={dev_uas:.4f} LAS={dev_las:.4f}"
        )

        if dev_las > best_las:
            best_las = dev_las
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    # ======================= TEST =======================
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    model.eval()
    with torch.no_grad():
        test_uas, test_las = evaluate(model, test_ld, device)

    writer.add_scalar("test/UAS", test_uas)
    writer.add_scalar("test/LAS", test_las)
    writer.close()

    print(f"TEST: UAS={test_uas:.4f} LAS={test_las:.4f}")


# ======================= MAIN =======================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset/UD_Vietnamese-VTB")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--freeze_epochs", type=int, default=8)
    ap.add_argument("--lr_phobert", type=float, default=2e-5)
    ap.add_argument("--lr_other", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--log_dir", default="runs/phobert_ud")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--mlp_size", type=int, default=512)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--arc_weight", type=float, default=1.0)
    ap.add_argument("--rel_weight", type=float, default=1.0)
    args = ap.parse_args()

    train(args)
