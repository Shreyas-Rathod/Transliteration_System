import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── Configuration ────────────────────────────────────────────

# Data paths (update if needed)
TRAIN_PATH = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV_PATH   = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"
TEST_PATH  = "/kaggle/input/gu-dataset/gu.translit.sampled.test.tsv"

# Best hyperparameters (from your Q5 sweep)
EMB_DIM    = 64
HID_DIM    = 128
DROPOUT    = 0.2
LR         = 1e-3
EPOCHS     = 5
BATCH_SIZE = 32

# Output HTML file
OUT_HTML = "attention_viz.html"

# ─── Device ────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Data Utilities ────────────────────────────────────────────

import pandas as pd
def read_data(path):
    df = pd.read_csv(path, sep="\t", header=None).iloc[:, :2].dropna()
    return [(str(a), str(b)) for a, b in zip(df[0], df[1])]

def build_vocab(pairs):
    src_chars, tgt_chars = set(), set()
    for s,t in pairs:
        src_chars.update(s)
        tgt_chars.update(t)
    src_vocab = {c:i+2 for i,c in enumerate(sorted(src_chars))}
    tgt_vocab = {c:i+2 for i,c in enumerate(sorted(tgt_chars))}
    src_vocab["<pad>"] = 0; src_vocab["<sos>"] = 1
    tgt_vocab["<pad>"] = 0; tgt_vocab["<sos>"] = 1
    return src_vocab, tgt_vocab

class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.sv = src_vocab
        self.tv = tgt_vocab
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s,t = self.data[idx]
        src_ids = [self.sv.get(c,1) for c in s]
        tgt_ids = [self.tv["<sos>"]] + [self.tv.get(c,1) for c in t]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts

# ─── Model Definition ─────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, hidden, enc_out):
        # hidden: [1,B,H], enc_out: [B,S,H]
        B,S,H = enc_out.size()
        h = hidden.squeeze(0).unsqueeze(1).repeat(1,S,1)   # [B,S,H]
        energy = torch.tanh(self.attn(torch.cat([h, enc_out], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)  # [B,S]

class AttnDecoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.emb   = nn.Embedding(out_dim, emb_dim, padding_idx=0)
        self.attn  = Attention(hid_dim)
        self.rnn   = nn.LSTM(emb_dim+hid_dim, hid_dim, batch_first=True)
        self.fc    = nn.Linear(emb_dim+hid_dim*2, out_dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, inp, hidden, cell, enc_out):
        # inp: [B], hidden/cell: [1,B,H], enc_out: [B,S,H]
        inp = inp.unsqueeze(1)                 # [B,1]
        emb = self.drop(self.emb(inp))         # [B,1,E]
        a   = self.attn(hidden, enc_out)       # [B,S]
        w   = a.unsqueeze(1).bmm(enc_out)      # [B,1,H]
        rnn_in = torch.cat([emb, w], dim=2)    # [B,1,E+H]
        out, (h,c) = self.rnn(rnn_in, (hidden,cell))
        out = out.squeeze(1); w = w.squeeze(1); emb = emb.squeeze(1)
        pred = self.fc(torch.cat([out, w, emb], dim=1))
        return pred, h, c, a

class Seq2SeqAtt(nn.Module):
    def __init__(self, emb_dim, hid_dim, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.decoder = AttnDecoder(emb_dim, hid_dim, tgt_vocab_size, dropout)
    def forward(self, src, tgt):
        # src=[B,S], tgt=[B,T]
        enc = self.enc_emb(src)
        enc_out, (h,c) = self.encoder(enc)
        B,T = tgt.size()
        outputs    = torch.zeros(B, T, self.decoder.fc.out_features).to(src.device)
        attentions = torch.zeros(B, T, src.size(1)).to(src.device)
        inp = tgt[:,0]
        for t in range(1,T):
            pred, h, c, a = self.decoder(inp, h, c, enc_out)
            outputs[:,t]    = pred
            attentions[:,t] = a
            inp = tgt[:,t]
        return outputs, attentions

# ─── Load & Prepare Data ───────────────────────────────────────

print("Loading data...")
train = read_data(TRAIN_PATH)
dev   = read_data(DEV_PATH)
test  = read_data(TEST_PATH)

src_vocab, tgt_vocab = build_vocab(train+dev)
inv_tgt_vocab = {i:c for c,i in tgt_vocab.items()}

train_loader = DataLoader(
    TransliterationDataset(train+dev, src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    TransliterationDataset(test, src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# ─── Train Model on train+dev ─────────────────────────────────

print("Training model on train+dev...")
model = Seq2SeqAtt(EMB_DIM, HID_DIM, len(src_vocab), len(tgt_vocab), DROPOUT).to(device)
opt   = optim.Adam(model.parameters(), lr=LR)
crit  = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src,tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out, _ = model(src, tgt[:,:-1])
        loss = crit(out.reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f" Epoch {epoch}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

# ─── Collect Attention Weights on Test ────────────────────────

print("Collecting attention weights on test set...")
all_attns = []
all_inputs = []
all_preds  = []

model.eval()
with torch.no_grad():
    for src, tgt in test_loader:
        src,tgt = src.to(device), tgt.to(device)
        out, attn = model(src, tgt)
        preds = out.argmax(-1).cpu().numpy()
        # store per-example
        for i in range(src.size(0)):
            s_str = "".join(inv_tgt_vocab[idx] for idx in src[i].cpu().numpy() if idx!=0)
            p_str = "".join(inv_tgt_vocab[idx] for idx in preds[i] if idx!=0)
            all_inputs.append(s_str)
            all_preds.append(p_str)
            all_attns.append(attn[i].cpu().numpy())

all_attns = np.stack(all_attns, axis=0)  # shape (N_test, T_out, T_in)

# ─── Generate Interactive HTML ────────────────────────────────

print(f"Building HTML → {OUT_HTML} ...")
html = ["<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Attention Connectivity</title>",
        "<style>",
        " body{font-family:sans-serif;padding:20px;} .sample{margin-bottom:2em;}",
        " .charspan{font-family:monospace;display:inline-block;padding:4px;}",
        " .output .charspan:hover{background:#eee;cursor:pointer;text-decoration:underline;}",
        "</style>",
        "</head><body>",
        "<h1>Seq2Seq Attention Connectivity</h1>",
        "<p>Hover over an <strong>output</strong> char to highlight <strong>input</strong> chars.</p>"]

for i,(src_str,p_str) in enumerate(zip(all_inputs, all_preds)):
    W = all_attns[i].tolist()
    html.append(f"<div class='sample' id='s{i}'>")
    html.append(f"<h3>Sample {i+1}</h3>")
    html.append("<div><strong>Input:</strong> ")
    html.extend(f"<span class='charspan'>{c}</span>" for c in src_str)
    html.append("</div><div><strong>Output:</strong> ")
    html.extend(f"<span class='charspan'>{c}</span>" for c in p_str)
    html.append("</div>")
    # embed script
    html.append("<script>(function(){")
    html.append(f"  var W={json.dumps(W)};")
    html.append(f"  var inp=document.querySelectorAll('#s{i} .sample .charspan, #s{i} .input .charspan'),"
                "out=document.querySelectorAll('#s{i} .output .charspan');")
    html.append("  out.forEach((span,t)=>{")
    html.append("    span.addEventListener('mouseover',()=>{")
    html.append("      var row=W[t];var m=Math.max(...row);")
    html.append("      row.forEach((w,s)=>{")
    html.append("        var alpha=m>0?w/m:0;")
    html.append("        inp[s].style.backgroundColor='rgba(0,255,0,'+alpha+')';")
    html.append("        inp[s].title=w.toFixed(3);")
    html.append("      });")
    html.append("    });")
    html.append("    span.addEventListener('mouseout',()=>{")
    html.append("      inp.forEach(e=>{e.style.backgroundColor=''; e.title='';});")
    html.append("    });")
    html.append("  });})();</script>")
    html.append("</div>")

html.append("</body></html>")

with open(OUT_HTML,"w",encoding="utf-8") as f:
    f.write("\n".join(html))

print("✅ Saved interactive visualization →", OUT_HTML)
