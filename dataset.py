
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

def read_conllu(path):
    sents = []
    words, pos, heads, rels = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sents.append((words, pos, heads, rels))
                words, pos, heads, rels = [], [], [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0]:
                continue
            words.append(cols[1])
            pos.append(cols[3])
            heads.append(int(cols[6]))
            rels.append(cols[7].split(":")[0])  
    return sents


class DependencyDataset(Dataset):
    def __init__(self, sents, pos2id, rel2id):
        self.data = []
        for w,p,h,r in sents:
            self.data.append((
                w,
                torch.tensor([pos2id[x] for x in p]),
                torch.tensor(h),
                torch.tensor([rel2id.get(x, rel2id["<unk>"]) for x in r])
            ))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
