"""The extraction model: any Hugging Face backbone plus three small heads.

For each field, the hidden state at its query marker (q) is compared with the
hidden state of every document token (h_t):

* pointer head  start_t = <W_s q, h_t>, end_t = <W_e q, h_t>, plus one learned
  "absent" logit from q, prepended to both distributions. Softmax over
  [absent] + tokens gives p_null directly (SQuAD 2.0 style).
* BIO head      three projections of q score each token as Outside / Begin /
  Inside, which yields any number of spans for `list` fields.

No tokens are generated: one forward pass scores every field at once.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import IGNORE, Feature

CONFIG_NAME = "extractor.json"
HEADS_NAME = "heads.safetensors"
FORMAT_VERSION = 1


class ExtractorModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.norm = nn.LayerNorm(hidden)
        self.q_start = nn.Linear(hidden, hidden)
        self.q_end = nn.Linear(hidden, hidden)
        self.q_null = nn.Linear(hidden, 2)
        self.q_bio = nn.Linear(hidden, 3 * hidden)
        self.scale = hidden ** -0.5

    def head_parameters(self):
        for module in (self.norm, self.q_start, self.q_end, self.q_null, self.q_bio):
            yield from module.parameters()

    @staticmethod
    def _per_field(proj: torch.Tensor, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Score proj[n] against every token of its own sequence -> [N, L]."""
        bsz, seq, hid = h.shape
        scores = proj @ h.reshape(bsz * seq, hid).T          # [N, B*L]
        scores = scores.view(-1, bsz, seq)                   # [N, B, L]
        return scores[torch.arange(scores.shape[0], device=h.device), batch_idx]

    def forward(self, input_ids, attention_mask, doc_mask, query_batch, query_pos):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.norm(out.last_hidden_state)
        q = h[query_batch, query_pos]                        # [N, H]
        dm = doc_mask[query_batch]                           # [N, L]
        neg = torch.finfo(h.dtype).min / 2

        start = self._per_field(self.q_start(q), h, query_batch) * self.scale
        end = self._per_field(self.q_end(q), h, query_batch) * self.scale
        start = start.masked_fill(~dm, neg)
        end = end.masked_fill(~dm, neg)
        null = self.q_null(q)
        start_logits = torch.cat([null[:, :1], start], dim=1)  # [N, 1+L]
        end_logits = torch.cat([null[:, 1:], end], dim=1)

        n = q.shape[0]
        bio_q = self.q_bio(q).view(n, 3, self.hidden)
        bio = torch.stack(
            [self._per_field(bio_q[:, k], h, query_batch) for k in range(3)], dim=-1
        ) * self.scale                                       # [N, L, 3]
        return {"start": start_logits.float(), "end": end_logits.float(), "bio": bio.float()}

    @staticmethod
    def loss(out, batch):
        # A loss over labels that are all IGNORE is NaN, so each term only
        # counts when the batch holds fields of its kind.
        total = out["start"].new_zeros(())
        if (batch["start_labels"] != IGNORE).any():
            total = total + 0.5 * (
                F.cross_entropy(out["start"], batch["start_labels"], ignore_index=IGNORE)
                + F.cross_entropy(out["end"], batch["end_labels"], ignore_index=IGNORE)
            )
        if (batch["bio_labels"] != IGNORE).any():
            total = total + F.cross_entropy(
                out["bio"].reshape(-1, 3), batch["bio_labels"].reshape(-1), ignore_index=IGNORE
            )
        return total

    # -- persistence --------------------------------------------------------

    def save(self, out_dir: str, tokenizer, extra: dict | None = None) -> None:
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        self.backbone.save_pretrained(os.path.join(out_dir, "backbone"))
        tokenizer.save_pretrained(os.path.join(out_dir, "backbone"))
        heads = {
            k: v.detach().cpu().contiguous()
            for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }
        save_file(heads, os.path.join(out_dir, HEADS_NAME))
        config = {"format_version": FORMAT_VERSION, "hidden": self.hidden, **(extra or {})}
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as fh:
            json.dump(config, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str, device="cpu", dtype=None):
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        with open(os.path.join(model_dir, CONFIG_NAME)) as fh:
            config = json.load(fh)
        bdir = os.path.join(model_dir, "backbone")
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        backbone = AutoModel.from_pretrained(bdir, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(bdir)
        model = cls(backbone, config["hidden"])
        missing, unexpected = model.load_state_dict(load_file(os.path.join(model_dir, HEADS_NAME)), strict=False)
        missing = [k for k in missing if not k.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"bad heads file: missing={missing} unexpected={unexpected}")
        if dtype is not None:
            model = model.to(dtype)
        return model.to(device).eval(), tokenizer, config


def build(backbone_name: str, **kwargs):
    """A fresh model on top of a pretrained (or local) backbone."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    backbone = AutoModel.from_pretrained(backbone_name, **kwargs)
    hidden = backbone.config.hidden_size
    return ExtractorModel(backbone, hidden), tokenizer


def collate(features: list[Feature], pad_id: int, device=None) -> dict:
    """Pad a list of features into tensors; fields are flattened across the batch."""
    seq = max(len(f.input_ids) for f in features)
    bsz = len(features)
    ids = torch.full((bsz, seq), pad_id, dtype=torch.long)
    att = torch.zeros((bsz, seq), dtype=torch.long)
    doc = torch.zeros((bsz, seq), dtype=torch.bool)
    qb, qp, sl, el, bio = [], [], [], [], []
    labelled = features[0].start_labels is not None
    for b, f in enumerate(features):
        n = len(f.input_ids)
        ids[b, :n] = torch.tensor(f.input_ids)
        att[b, :n] = 1
        doc[b, f.doc_start : f.doc_end] = True
        for i, pos in enumerate(f.query_pos):
            qb.append(b)
            qp.append(pos)
            if labelled:
                sl.append(f.start_labels[i])
                el.append(f.end_labels[i])
                tags = f.bio_labels[i] + [IGNORE] * (seq - n)
                bio.append(tags)
    batch = {
        "input_ids": ids,
        "attention_mask": att,
        "doc_mask": doc,
        "query_batch": torch.tensor(qb, dtype=torch.long),
        "query_pos": torch.tensor(qp, dtype=torch.long),
    }
    if labelled:
        batch["start_labels"] = torch.tensor(sl, dtype=torch.long)
        batch["end_labels"] = torch.tensor(el, dtype=torch.long)
        batch["bio_labels"] = torch.tensor(bio, dtype=torch.long)
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def model_inputs(batch: dict) -> dict:
    return {k: batch[k] for k in ("input_ids", "attention_mask", "doc_mask", "query_batch", "query_pos")}
