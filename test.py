#!/usr/bin/env python3
import os
import collections
import torch
from torch.utils.data import DataLoader

from model import PhoBERTParser
from dataset import read_conllu, DependencyDataset
from evaluate import evaluate
from train import collate_fn  

# ======================
# CONFIG
# ======================
data_dir = "dataset/UD_Vietnamese-VTB"
ckpt_path = "checkpoints/best_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PRINT = 50
OUTPUT_FILE = "error_analysis.txt"

# bucket config
LEN_BUCKET = 10
DIST_BUCKETS = [(0,1), (2,3), (4,6), (7,100)]

# ======================
# LOAD DATA
# ======================
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

# ======================
# LOAD MODEL
# ======================
model = PhoBERTParser(
    len(pos2id),
    len(rel2id),
    hidden=400,
    mlp_size=512
).to(device)

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

uas, las = evaluate(model, test_ld, device)

# ======================
# ERROR ANALYSIS
# ======================
@torch.no_grad()
def analyze_errors(model, loader):
    errors = []

    pos_stats  = collections.defaultdict(lambda: {"total":0,"head":0,"rel":0})
    len_stats  = collections.defaultdict(lambda: {"total":0,"head":0,"rel":0})
    dist_stats = collections.defaultdict(lambda: {"total":0,"head":0,"rel":0})
    rel_stats  = collections.defaultdict(lambda: {"total":0,"head":0,"rel":0})

    root_wrong = 0
    punct_wrong = 0

    sent_idx = 0

    for words, pos, heads, rels, mask in loader:
        pos, heads, rels, mask = pos.to(device), heads.to(device), rels.to(device), mask.to(device)

        arc_scores, rel_scores = model(words, pos)
        pred_heads = arc_scores.argmax(dim=-1)

        B, L = pred_heads.shape
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        i_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pred_rels = rel_scores[b_idx, i_idx, pred_heads].argmax(dim=-1)

        valid = mask & (heads >= 0)
        sent_len = valid.sum().item()
        len_bucket = (sent_len // LEN_BUCKET) * LEN_BUCKET

        for i in range(L):
            if not valid[0, i]:
                continue

            gold_h = heads[0,i].item()
            gold_r = rels[0,i].item()
            pred_h = pred_heads[0,i].item()
            pred_r = pred_rels[0,i].item()

            pos_tag = id2pos[pos[0,i].item()]
            rel_tag = id2rel[gold_r]

            head_ok = (gold_h == pred_h)
            rel_ok  = (gold_r == pred_r)

            # POS
            pos_stats[pos_tag]["total"] += 1
            if not head_ok:
                pos_stats[pos_tag]["head"] += 1
            if head_ok and not rel_ok:
                pos_stats[pos_tag]["rel"] += 1

            # Sentence length
            len_stats[len_bucket]["total"] += 1
            if not head_ok:
                len_stats[len_bucket]["head"] += 1
            if head_ok and not rel_ok:
                len_stats[len_bucket]["rel"] += 1

            # Dependency distance
            dist = abs(i - gold_h)
            for lo, hi in DIST_BUCKETS:
                if lo <= dist <= hi:
                    key = f"{lo}-{hi}"
                    dist_stats[key]["total"] += 1
                    if not head_ok:
                        dist_stats[key]["head"] += 1
                    if head_ok and not rel_ok:
                        dist_stats[key]["rel"] += 1
                    break

            # Relation
            rel_stats[rel_tag]["total"] += 1
            if not head_ok:
                rel_stats[rel_tag]["head"] += 1
            if head_ok and not rel_ok:
                rel_stats[rel_tag]["rel"] += 1

            # Root / punctuation
            if gold_h == 0 and pred_h != 0:
                root_wrong += 1
            if pos_tag == "PUNCT" and not head_ok:
                punct_wrong += 1

            # qualitative
            if (not head_ok or not rel_ok) and len(errors) < MAX_PRINT:
                errors.append({
                    "sent_idx": sent_idx,
                    "sent": " ".join(words[0]),
                    "token": words[0][i],
                    "pos": pos_tag,
                    "gold": (gold_h, rel_tag),
                    "pred": (pred_h, id2rel[pred_r]),
                    "type": "HEAD" if not head_ok else "REL"
                })

        sent_idx += 1

    return errors, pos_stats, len_stats, dist_stats, rel_stats, root_wrong, punct_wrong


results = analyze_errors(model, test_ld)

# ======================
# WRITE OUTPUT
# ======================
with open(OUTPUT_FILE, "w", encoding="utf8") as f:
    f.write(f"UAS={uas:.4f}, LAS={las:.4f}\n\n")

    errors, pos_s, len_s, dist_s, rel_s, root_w, punct_w = results

    f.write("=== QUALITATIVE ERRORS ===\n")
    for e in errors:
        f.write(f"[Sent {e['sent_idx']}] {e['sent']}\n")
        f.write(f" token='{e['token']}' POS={e['pos']}\n")
        f.write(f" gold={e['gold']} pred={e['pred']} TYPE={e['type']}\n\n")

    def dump_table(title, stats):
        f.write(f"\n=== {title} ===\n")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]["total"]):
            if v["total"] == 0:
                continue
            f.write(f"{str(k):>10s} | total={v['total']:4d} | "
                    f"head_err={v['head']/v['total']:.3f} | "
                    f"rel_err={v['rel']/v['total']:.3f}\n")

    dump_table("ERROR BY POS", pos_s)
    dump_table("ERROR BY SENTENCE LENGTH", len_s)
    dump_table("ERROR BY DEPENDENCY DISTANCE", dist_s)
    dump_table("ERROR BY RELATION TYPE", rel_s)

    f.write(f"\nROOT WRONG: {root_w}\n")
    f.write(f"PUNCT HEAD WRONG: {punct_w}\n")

print(f"[DONE] Error analysis saved to {OUTPUT_FILE}")
