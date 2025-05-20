# ───────────── Question 5(d): Attention Heatmaps without checkpoint file ─────────────

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1) Force CPU if you hit NCCL issues
# import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Data utilities (reuse from your Q2/Q5 cells)

def read_data(path):
    df = pd.read_csv(path, sep="\t", header=None).iloc[:, :2].dropna()
    return [(str(a), str(b)) for a,b in zip(df[0], df[1])]

def build_vocab(data):
    s,t = set(), set()
    for a,b in data:
        s.update(a); t.update(b)
    sv = {c:i+2 for i,c in enumerate(sorted(s))}
    tv = {c:i+2 for i,c in enumerate(sorted(t))}
    sv['<pad>']=0; sv['<sos>']=1
    tv['<pad>']=0; tv['<sos>']=1
    return sv, tv

class TransliterationDataset(Dataset):
    def __init__(self,data,src_vocab,tgt_vocab):
        self.data,self.sv,self.tv=data,src_vocab,tgt_vocab
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        a,b = self.data[i]
        src = [self.sv.get(c,1) for c in a]
        tgt = [self.tv['<sos>']] + [self.tv.get(c,1) for c in b]
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts

# 3) Attention model classes (reuse from Q5)

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, hidden, enc_out):
        # hidden: [1,B,H] → [B,H]; enc_out: [B,S,H]
        B, S, H = enc_out.size()
        h = hidden.squeeze(0).unsqueeze(1).repeat(1,S,1)   # [B,S,H]
        energy = torch.tanh(self.attn(torch.cat([h, enc_out], dim=2)))  # [B,S,H]
        return torch.softmax(self.v(energy).squeeze(2), dim=1)         # [B,S]

class AttnDecoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.emb     = nn.Embedding(out_dim, emb_dim, padding_idx=0)
        self.attn    = Attention(hid_dim)
        self.rnn     = nn.LSTM(emb_dim+hid_dim, hid_dim, batch_first=True)
        self.fc      = nn.Linear(emb_dim+hid_dim*2, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, cell, enc_out):
        # input: [B], hidden/cell: [1,B,H], enc_out: [B,S,H]
        inp = input.unsqueeze(1)                # [B,1]
        emb = self.dropout(self.emb(inp))       # [B,1,emb_dim]
        a   = self.attn(hidden, enc_out)        # [B,S]
        w   = a.unsqueeze(1).bmm(enc_out)       # [B,1,H]
        rnn_in = torch.cat([emb, w], dim=2)     # [B,1,emb+H]
        out, (h,c) = self.rnn(rnn_in, (hidden,cell))
        out = out.squeeze(1); w = w.squeeze(1); emb = emb.squeeze(1)
        pred = self.fc(torch.cat([out, w, emb], dim=1))  # [B,out_dim]
        return pred, h, c, a

class Seq2SeqAtt(nn.Module):
    def __init__(self, emb_dim, hid_dim, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.decoder = AttnDecoder(emb_dim, hid_dim, tgt_vocab_size, dropout)
    def forward(self, src, tgt):
        # src: [B, S]; tgt: [B, T]
        enc_in  = self.enc_emb(src) 
        enc_out, (h, c) = self.encoder(enc_in)  # enc_out: [B,S,H]
        B, T = tgt.size()
        outputs = torch.zeros(B, T, self.decoder.fc.out_features).to(src.device)
        attentions = torch.zeros(B, T, src.size(1)).to(src.device)
        inp = tgt[:,0]
        for t in range(1, T):
            pred, h, c, a = self.decoder(inp, h, c, enc_out)
            outputs[:,t] = pred
            attentions[:,t] = a
            inp = tgt[:,t]
        return outputs, attentions

# 4) Paths and load data
TRAIN = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV   = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"
TEST  = "/kaggle/input/gu-dataset/gu.translit.sampled.test.tsv"

train_data = read_data(TRAIN)
dev_data   = read_data(DEV)
test_data  = read_data(TEST)

src_vocab, tgt_vocab = build_vocab(train_data+dev_data)
inv_tgt_vocab = {i:c for c,i in tgt_vocab.items()}

# 5) Dataloaders
train_loader = DataLoader(
    TransliterationDataset(train_data+dev_data, src_vocab, tgt_vocab),
    batch_size=32, shuffle=True, collate_fn=collate_fn
)

# 6) Initialize & retrain in-memory with your best hyperparams
best = {"emb_dim":64, "hid_dim":128, "dropout":0.2}
model = Seq2SeqAtt(
    emb_dim = best["emb_dim"],
    hid_dim = best["hid_dim"],
    src_vocab_size = len(src_vocab),
    tgt_vocab_size = len(tgt_vocab),
    dropout = best["dropout"]
).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(5):
    model.train()
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out, _ = model(src, tgt)
        loss = crit(out[:,1:].reshape(-1, out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward(); opt.step()

# 7) Plot 3×3 grid of attention heatmaps on the first 9 test samples
fig, axes = plt.subplots(3, 3, figsize=(9,9))
for idx, (src_str, tgt_str) in enumerate(test_data[:9]):
    src_ids = torch.tensor([[src_vocab.get(c,1) for c in src_str]]).to(device)
    tgt_ids = torch.tensor([[tgt_vocab['<sos>']] + [tgt_vocab.get(c,1) for c in tgt_str]]).to(device)
    with torch.no_grad():
        _, attn = model(src_ids, tgt_ids)
    attn_mat = attn[0,1:len(tgt_str)+1].cpu().numpy()  # [tgt_len, src_len]

    ax = axes[idx//3, idx%3]
    im = ax.imshow(attn_mat, aspect='auto')
    ax.set_title(src_str)
    ax.set_xticks(range(len(src_str)))
    ax.set_xticklabels(list(src_str), rotation=90)
    ax.set_yticks(range(len(tgt_str)))
    ax.set_yticklabels(list(tgt_str))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
