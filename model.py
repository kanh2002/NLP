
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional


class Biaffine(nn.Module):
    def __init__(self, in_dim: int, out: int = 1):
        super().__init__()
       
        self.U = nn.Parameter(torch.empty(out, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       
        B, L, D = x.size()
        ones = x.new_ones(B, L, 1)
        x_ = torch.cat([x, ones], dim=-1)  # (B, L, D+1)
        y_ = torch.cat([y, ones], dim=-1)
        s = torch.einsum("bxi, oij, byj -> boxy", x_, self.U, y_)
        if self.U.size(0) == 1:
            return s.squeeze(1)  # (B, L, L)
        else:
            return s.permute(0, 2, 3, 1)  # (B, L, L, out)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.33):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim, eps=1e-6)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.ln(x)
        return x


class PhoBERTParser(nn.Module):
    

    def __init__(
        self,
        pos_size: int,
        rel_size: int,
        hidden: int = 512,
        mlp_size: int = 768,
        phobert_name: str = "vinai/phobert-base",
        pos_emb_size: int = 64,
        lstm_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
       
        self.tokenizer = AutoTokenizer.from_pretrained(phobert_name, use_fast=False)
        self.phobert = AutoModel.from_pretrained(phobert_name)

        phobert_hidden = getattr(self.phobert.config, "hidden_size", 768)

        self.pos_emb = nn.Embedding(pos_size, pos_emb_size, padding_idx=0)

        self.lstm = nn.LSTM(
            phobert_hidden + pos_emb_size,
            hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        mlp_in = hidden * 2
        self.mlp_arc_dep = MLP(mlp_in, mlp_size, dropout=dropout)
        self.mlp_arc_head = MLP(mlp_in, mlp_size, dropout=dropout)
        self.mlp_rel_dep = MLP(mlp_in, mlp_size, dropout=dropout)
        self.mlp_rel_head = MLP(mlp_in, mlp_size, dropout=dropout)

        self.arc_biaff = Biaffine(mlp_size, out=1)
        self.rel_biaff = Biaffine(mlp_size, out=rel_size)

    # ---------------- Encoding ----------------
    def encode_batch(self, batch_words: List[List[str]], max_len: int, device: torch.device) -> torch.Tensor:
      
        B = len(batch_words)
        
        sentences = [" ".join(words) for words in batch_words]

        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

      
        enc_device = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        outputs = self.phobert(**enc_device)
        hidden = outputs.last_hidden_state  # (B, T, H)
        B_enc, T, H = hidden.size()

       
        word_token_lens_batch: List[List[int]] = []
        for words in batch_words:
            lens = []
            for w in words:
                toks = self.tokenizer(w, add_special_tokens=False, return_tensors=None)
                tok_ids = toks.get("input_ids", [])
                lens.append(len(tok_ids))
            word_token_lens_batch.append(lens)

        all_special_ids = set(self.tokenizer.all_special_ids or [])

        word_embs_batch = []
        input_ids = enc["input_ids"]  # (B, T) on CPU but we only inspect ids
        for b in range(B):
            sent_ids = input_ids[b].tolist()
            w_lens = word_token_lens_batch[b]
            sent_embs = []
            
            ptr = 0
           
            while ptr < T and sent_ids[ptr] in all_special_ids:
                ptr += 1

            for w_i, w_len in enumerate(w_lens[:max_len]):
                if w_len <= 0:
                
                    sent_embs.append(hidden.new_zeros(H))
                    continue

            
                pos_list = []
                while ptr < T and len(pos_list) < w_len:
                    if sent_ids[ptr] not in all_special_ids:
                        pos_list.append(ptr)
                    ptr += 1

               
                if len(pos_list) == 0:
                    sent_embs.append(hidden.new_zeros(H))
                else:
                    tok_idxs = torch.tensor(pos_list, device=device)
                    emb = hidden[b, tok_idxs].mean(dim=0)
                    sent_embs.append(emb)

                
                while ptr < T and sent_ids[ptr] in all_special_ids:
                    ptr += 1

            # pad / truncate to max_len
            if len(sent_embs) < max_len:
                pad_count = max_len - len(sent_embs)
                pad = hidden.new_zeros(pad_count, H)
                if len(sent_embs) == 0:
                    sent_embs = pad
                else:
                    sent_embs = torch.stack(sent_embs, dim=0)
                    sent_embs = torch.cat([sent_embs, pad], dim=0)
            else:
                sent_embs = torch.stack(sent_embs[:max_len], dim=0)

            word_embs_batch.append(sent_embs)

        return torch.stack(word_embs_batch, dim=0)  # (B, max_len, H)

    # ---------------- Forward ----------------
    def forward(
        self,
        batch_words: List[List[str]],
        pos_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - batch_words: List[List[str]] (surface words)
        - pos_ids: (B, L) tensor of POS ids (0 is padding)
        - mask: optional (B, L) boolean; if None, computed as pos_ids != 0
        returns:
        - arc_scores: (B, L, L)
        - rel_scores: (B, L, L, R)
        """
        device = next(self.parameters()).device
        B, L = pos_ids.size()
        if mask is None:
            mask = pos_ids != 0

        word_embs = self.encode_batch(batch_words, max_len=L, device=device)  # (B, L, H)
        pos_embs = self.pos_emb(pos_ids.to(device))
        x = torch.cat([word_embs, pos_embs], dim=-1)  # (B, L, phobertH + pos_emb)
        h, _ = self.lstm(x)
        h = self.dropout(h)

        arc_dep = self.mlp_arc_dep(h)
        arc_head = self.mlp_arc_head(h)
        rel_dep = self.mlp_rel_dep(h)
        rel_head = self.mlp_rel_head(h)

        arc_scores = self.arc_biaff(arc_dep, arc_head)  # (B, L, L)
        rel_scores = self.rel_biaff(rel_dep, rel_head)  # (B, L, L, R)
        return arc_scores, rel_scores

    # ---------------- Decoding ----------------
    @staticmethod
    def chu_liu_edmonds(scores: torch.Tensor) -> List[int]:
        """
        Chu-Liu-Edmonds for maximum spanning tree (expects numpy array (L,L))
        returns list of head indices length L (head for each token)
        """
        import numpy as np

        scores = scores.copy()
        L = scores.shape[0]
        if L > 200:
            return list(np.argmax(scores, axis=0))

        heads = list(np.argmax(scores, axis=0))

        for i in range(L):
            if heads[i] == i:
                col = scores[:, i]
                idxs = np.argsort(col)
                heads[i] = idxs[-2] if len(idxs) > 1 else idxs[-1]

        def find_cycle(hds):
            visited = [0] * L
            stack = [0] * L
            for i in range(L):
                v = i
                path = []
                while not visited[v]:
                    visited[v] = 1
                    stack[v] = 1
                    path.append(v)
                    v = hds[v]
                    if v < 0 or v >= L:
                        break
                    if stack[v]:
                        cyc = []
                        cur = v
                        while True:
                            cyc.append(cur)
                            cur = hds[cur]
                            if cur == v:
                                break
                        return cyc
                for p in path:
                    stack[p] = 0
            return None

        cyc = find_cycle(heads)
        while cyc is not None:
            best_node = None
            best_improve = None
            new_head = None
            for node in cyc:
                col = scores[:, node]
                sorted_idx = np.argsort(col)
                best = sorted_idx[-1]
                second = sorted_idx[-2] if len(sorted_idx) > 1 else best
                diff = col[best] - col[second]
                if best_node is None or diff < best_improve:
                    best_improve = diff
                    best_node = node
                    new_head = second
            heads[best_node] = new_head
            if heads[best_node] == best_node:
                col = scores[:, best_node]
                idxs = np.argsort(col)
                heads[best_node] = idxs[-2] if len(idxs) > 1 else idxs[-1]
            cyc = find_cycle(heads)
        return heads

    def decode(self, arc_scores: torch.Tensor, rel_scores: torch.Tensor, mask: torch.Tensor, use_mst: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        arc_scores: (B, L, L)
        rel_scores: (B, L, L, R)
        mask: (B, L) bool
        returns:
        - heads_out: (B, L) long
        - rels_out: (B, L) long
        """
        B, L, _ = arc_scores.shape
        device = arc_scores.device
        heads_out = torch.zeros(B, L, dtype=torch.long, device=device)
        rels_out = torch.zeros(B, L, dtype=torch.long, device=device)

        for b in range(B):
            valid_len = int(mask[b].sum().item())
            if valid_len == 0:
                continue
            scores = arc_scores[b][:valid_len, :valid_len].detach().cpu().numpy()
            if use_mst:
                heads = self.chu_liu_edmonds(scores)
            else:
                heads = list(torch.argmax(arc_scores[b][:valid_len, :valid_len], dim=1).cpu().numpy())

            # pad/truncate to L
            if len(heads) < L:
                heads = heads + [0] * (L - len(heads))
            else:
                heads = heads[:L]
            heads_out[b, :L] = torch.tensor(heads, dtype=torch.long, device=device)

            for i in range(valid_len):
                h = heads[i]
                rel_logits = rel_scores[b, i, h]
                rels_out[b, i] = torch.argmax(rel_logits).to(device)
        return heads_out, rels_out

    # ---------------- Utilities ----------------
    def freeze_phobert(self, freeze: bool = True):
        for p in self.phobert.parameters():
            p.requires_grad = not (not freeze)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        loc = None if map_location is None else map_location
        self.load_state_dict(torch.load(path, map_location=loc))
