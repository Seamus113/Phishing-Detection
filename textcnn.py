import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
DATA_PATH = "dataset.jsonl" 
SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MAX_VOCAB = 50000          
MIN_FREQ = 2               
MAX_LEN = 400              
BATCH_SIZE = 64


EMBED_DIM = 128
NUM_FILTERS = 128
KERNEL_SIZES = (3, 4, 5)
DROPOUT = 0.5


EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

TOKEN_RE = re.compile(r"[a-z0-9]+")  





def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in rec or "label" not in rec:
                continue
            try:
                rec["label"] = int(rec["label"])
            except Exception:
                continue
            if rec["label"] not in (0, 1):
                continue
            data.append(rec)
    return data


def train_val_test_split(data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.shuffle(data)
    n = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int


def build_vocab(texts: List[str]) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    itos = ["<PAD>", "<UNK>"]  # 0,1
    words = [w for w, c in counter.items() if c >= MIN_FREQ]
    words.sort(key=lambda w: counter[w], reverse=True)
    words = words[: max(0, MAX_VOCAB - len(itos))]

    itos.extend(words)
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=0, unk_id=1)


def encode(text: str, vocab: Vocab) -> List[int]:
    ids = []
    for tok in tokenize(text)[:MAX_LEN]:
        ids.append(vocab.stoi.get(tok, vocab.unk_id))
    return ids


class TextDataset(Dataset):
    def __init__(self, records: List[Dict], vocab: Vocab):
        self.records = records
        self.vocab = vocab

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = encode(r["text"], self.vocab)
        y = int(r["label"])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_batch(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths) if lengths else 0

    padded = torch.full((len(xs), max_len), 0, dtype=torch.long)  # PAD=0
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.stack(ys)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_sizes: Tuple[int, ...], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        # x: [B, T]
        emb = self.embedding(x)       # [B, T, E]
        emb = emb.transpose(1, 2)     # [B, E, T]

        feats = []
        for conv in self.convs:
            h = torch.relu(conv(emb))        # [B, F, T-K+1]
            p = torch.max(h, dim=2).values   # [B, F]
            feats.append(p)

        out = torch.cat(feats, dim=1)        # [B, F * len(K)]
        out = self.dropout(out)
        logits = self.fc(out)                # [B, 2]
        return logits


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    loss_sum = 0.0
    correct = 0

    # confusion matrix for positive class=1
    tp = fp = fn = tn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()

    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_jsonl(DATA_PATH)
    if not data:
        raise RuntimeError(f"No records loaded from {DATA_PATH}")

    n0 = sum(1 for r in data if int(r["label"]) == 0)
    n1 = len(data) - n0
    print(f"Loaded: {len(data)} records | label0={n0} label1={n1}")

    train, val, test = train_val_test_split(data)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    vocab = build_vocab([r["text"] for r in train])
    print(f"Vocab size: {len(vocab.itos)} (MAX_VOCAB={MAX_VOCAB}, MIN_FREQ={MIN_FREQ})")

    train_ds = TextDataset(train, vocab)
    val_ds = TextDataset(val, vocab)
    test_ds = TextDataset(test, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    model = TextCNN(
        vocab_size=len(vocab.itos),
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = loss_sum / max(1, total)
        val_m = evaluate_metrics(model, val_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_acc={val_m['acc']:.4f} | "
            f"val_p={val_m['precision']:.4f} | "
            f"val_r={val_m['recall']:.4f} | "
            f"val_f1={val_m['f1']:.4f}"
        )

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = evaluate_metrics(model, test_loader, device)
    print(
        f"Best val_acc={best_val_acc:.4f} | "
        f"test_loss={test_m['loss']:.4f} | "
        f"test_acc={test_m['acc']:.4f} | "
        f"test_p={test_m['precision']:.4f} | "
        f"test_r={test_m['recall']:.4f} | "
        f"test_f1={test_m['f1']:.4f}"
    )
    print(
        f"Confusion (pos=1): TP={test_m['tp']} FP={test_m['fp']} "
        f"FN={test_m['fn']} TN={test_m['tn']}"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.itos,
            "config": {
                "MAX_LEN": MAX_LEN,
                "EMBED_DIM": EMBED_DIM,
                "NUM_FILTERS": NUM_FILTERS,
                "KERNEL_SIZES": KERNEL_SIZES,
                "DROPOUT": DROPOUT,
            },
        },
        "textcnn_checkpoint.pt",
    )
    print("Saved: textcnn_checkpoint.pt")


if __name__ == "__main__":
    main()

