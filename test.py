#!/usr/bin/env python3
import os
import collections
import torch
from torch.utils.data import DataLoader

from model import PhoBERTParser
from dataset import read_conllu, DependencyDataset
from evaluate import evaluate
from train import collate_fn  

data_dir = "dataset/UD_Vietnamese-VTB"
ckpt_path = "checkpoints/best_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PRINT = 50  # số lỗi in ra chi tiết


train_sents = read_conllu(os.path.join(data_dir, "vi_vtb-ud-train.conllu"))
test_sents  = read_conllu(os.path.join(data_dir, "vi_vtb-ud-test.conllu"))

pos2id = {"<pad>": 0}
rel2id = {"<unk>": 0}
for _, pos, _, rel in train_sents:
    for p in pos:
        pos2id.setdefault(p, len(pos2id))
    for r in rel:
        rel2id.setdefault(r, len(rel2id))

id2pos = {v: k for k, v in pos2id.items()}
id2rel = {v: k for k, v in rel2id.items()}

test_ds = DependencyDataset(test_sents, pos2id, rel2id)
test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)


model = PhoBERTParser(
    len(pos2id),
    len(rel2id),
    hidden=400,   
    mlp_size=512  
).to(device)

state = torch.load(ckpt_path, map_location=device)
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
# model.load_state_dict(state)
model.eval()

uas, las = evaluate(model, test_ld, device)
print(f"TEST (from checkpoint): UAS={uas:.4f}, LAS={las:.4f}")

#  Phân tích lỗi chi tiết
@torch.no_grad()
def analyze_errors(model, loader, device, id2pos, id2rel, max_examples=50):
    model.eval()
    errors = []
    pos_stats = collections.defaultdict(lambda: {"total": 0, "head_wrong": 0, "rel_wrong": 0})

    sent_idx = 0
    for words, pos, heads, rels, mask in loader:
        pos   = pos.to(device)
        heads = heads.to(device)
        rels  = rels.to(device)
        mask  = mask.to(device)

        arc_scores, rel_scores = model(words, pos)  # (1, L, L), (1, L, L, R)
        B, L, _ = arc_scores.shape

        # dự đoán head
        pred_heads = arc_scores.argmax(dim=-1)  # (B, L)

        # dự đoán nhãn tại head dự đoán
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        i_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pred_rels = rel_scores[b_idx, i_idx, pred_heads].argmax(dim=-1)  # (B, L)

        valid = mask & (heads >= 0)
        b = 0  # batch_size=1

        for i in range(L):
            if not valid[b, i]:
                continue

            gold_head = heads[b, i].item()
            gold_rel  = rels[b, i].item()
            pred_head = pred_heads[b, i].item()
            pred_rel  = pred_rels[b, i].item()
            pos_id    = pos[b, i].item()
            pos_tag   = id2pos.get(pos_id, "UNK")

            pos_stats[pos_tag]["total"] += 1
            head_correct = (pred_head == gold_head)
            rel_correct  = (pred_rel == gold_rel)

            if not head_correct:
                pos_stats[pos_tag]["head_wrong"] += 1
            if head_correct and not rel_correct:
                pos_stats[pos_tag]["rel_wrong"] += 1

            if (not head_correct) or (not rel_correct):
                if len(errors) < max_examples:
                    errors.append({
                        "sent_idx": sent_idx,
                        "words": words[b],
                        "token_idx": i,
                        "token": words[b][i],
                        "pos": pos_tag,
                        "gold_head": gold_head,
                        "pred_head": pred_head,
                        "gold_rel": id2rel.get(gold_rel, "UNK"),
                        "pred_rel": id2rel.get(pred_rel, "UNK"),
                        "head_correct": head_correct,
                        "rel_correct": rel_correct,
                    })
        sent_idx += 1

    return errors, pos_stats

errors, pos_stats = analyze_errors(model, test_ld, device, id2pos, id2rel, max_examples=MAX_PRINT)

print("\n=== Một số ví dụ lỗi (tối đa", MAX_PRINT, "token) ===")
for e in errors:
    sent = " ".join(e["words"])
    print(f"[Sent {e['sent_idx']}] {sent}")
    print(f"  token {e['token_idx']}='{e['token']}' POS={e['pos']}")
    print(f"    gold: head={e['gold_head']} rel={e['gold_rel']}")
    print(f"    pred: head={e['pred_head']} rel={e['pred_rel']}")
    status = []
    if not e["head_correct"]:
        status.append("HEAD_WRONG")
    if e["head_correct"] and not e["rel_correct"]:
        status.append("REL_WRONG_ONLY")
    print("   ", ", ".join(status))
    print()

print("=== Thống kê lỗi theo POS ===")
for pos_tag, st in sorted(pos_stats.items(), key=lambda x: x[1]["total"], reverse=True):
    if st["total"] == 0:
        continue
    head_err_rate = st["head_wrong"] / st["total"]
    rel_err_rate  = st["rel_wrong"] / st["total"]
    print(f"{pos_tag:>8s} | total={st['total']:4d} | "
          f"head_err={head_err_rate:.3f} | rel_err_given_head={rel_err_rate:.3f}")