# ─────────────── Question 5: Attention Seq2Seq ───────────────

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader

# ─── Add at the top, right after imports ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB login
wandb.login(key='b4b7e96266129ba04e848c1f16f53dbf520f6c16')

# os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_CONSOLE"] = "off"

# ── Reuse your data utils ─────────────────────────────────────
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
    return sv,tv

class TransliterationDataset(Dataset):
    def __init__(self,data,src_vocab,tgt_vocab):
        self.data,self.sv,self.tv=data,src_vocab,tgt_vocab
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        a,b=self.data[i]
        src=[self.sv.get(c,1) for c in a]
        tgt=[self.tv['<sos>']]+[self.tv.get(c,1) for c in b]
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts

# ── Attention modules ─────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch, hid_dim]
        # encoder_outputs = [batch, src_len, hid_dim]
        batch, src_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)    # [batch, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)                 # [batch, src_len]
        return F.softmax(attention, dim=1)

class AttnDecoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=0)
        self.attention = Attention(hid_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc   = nn.Linear(emb_dim + hid_dim*2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch]   ; hidden,cell = [1,batch,hid_dim]
        input = input.unsqueeze(1)                           # [batch,1]
        embedded = self.dropout(self.embedding(input))       # [batch,1,emb_dim]
        a = self.attention(hidden.squeeze(0), encoder_outputs)  # [batch,src_len]
        a = a.unsqueeze(1)                                   # [batch,1,src_len]
        weighted = torch.bmm(a, encoder_outputs)             # [batch,1,hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)   # [batch,1,emb+hid]
        output, (h, c) = self.rnn(rnn_input, (hidden, cell)) 
        # output=[batch,1,hid]; h,c=[1,batch,hid]
        embedded = embedded.squeeze(1)                       # [batch,emb_dim]
        output   = output.squeeze(1)                         # [batch,hid_dim]
        weighted = weighted.squeeze(1)                       # [batch,hid_dim]
        prediction = self.fc(torch.cat((output, weighted, embedded), dim=1))
        # [batch, out_dim]
        return prediction, h, c, a.squeeze(1)                # return attention weights

class Seq2SeqAtt(nn.Module):
    def __init__(self, emb_dim, hid_dim, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()
        self.encoder_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.decoder     = AttnDecoder(emb_dim, hid_dim, tgt_vocab_size, dropout)

    def forward(self, src, tgt):
        # src=[B,src_len], tgt=[B,tgt_len]
        enc_emb = self.encoder_emb(src)
        enc_out, (h, c) = self.encoder_rnn(enc_emb)
        batch, tgt_len = tgt.size()
        outputs = torch.zeros(batch, tgt_len, self.decoder.fc.out_features).to(src.device)
        attentions = torch.zeros(batch, tgt_len, src.size(1)).to(src.device)
        input_tok = tgt[:,0]  # <sos>
        for t in range(1, tgt_len):
            pred, h, c, a = self.decoder(input_tok, h, c, enc_out)
            outputs[:,t] = pred
            attentions[:,t] = a
            input_tok = tgt[:,t]  # teacher forcing
        return outputs, attentions

# ── Data prep ─────────────────────────────────────────────────
TRAIN = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV   = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"
TEST  = "/kaggle/input/gu-dataset/gu.translit.sampled.test.tsv"

train_data = read_data(TRAIN)
dev_data   = read_data(DEV)
test_data  = read_data(TEST)

src_vocab, tgt_vocab = build_vocab(train_data + dev_data)
inv_tgt_vocab = {i:c for c,i in tgt_vocab.items()}

train_loader = DataLoader(
    TransliterationDataset(train_data, src_vocab, tgt_vocab),
    batch_size=32, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    TransliterationDataset(dev_data, src_vocab, tgt_vocab),
    batch_size=32, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    TransliterationDataset(test_data, src_vocab, tgt_vocab),
    batch_size=32, shuffle=False, collate_fn=collate_fn
)

# ── WandB sweep: tune emb_dim, hid_dim, dropout, lr ───────────
def train_attn_model():
    run = wandb.init(project="CS24M046_DA6401_A3_t1", reinit=True)
    config = run.config

    model = Seq2SeqAtt(
        emb_dim    = config.emb_dim,
        hid_dim    = config.hid_dim,
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        dropout    = config.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, config.epochs + 1):          # ← use config.epochs
        # — Training —
        model.train()
        total_loss = correct = total = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output, _ = model(src, tgt[:, :-1])       # remember Seq2SeqAtt returns (outputs, attentions)
            loss = criterion(output.view(-1, output.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(-1)
            mask = (tgt[:,1:] != 0)
            correct += ((preds == tgt[:,1:]) * mask).sum().item()
            total   += mask.sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        # — Validation —
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for src, tgt in dev_loader:
                src, tgt = src.to(device), tgt.to(device)
                output, _ = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt[:,1:].reshape(-1))
                val_loss += loss.item()
                preds = output.argmax(-1)
                mask = (tgt[:,1:] != 0)
                val_correct += ((preds == tgt[:,1:]) * mask).sum().item()
                val_total   += mask.sum().item()

        val_loss_avg = val_loss / len(dev_loader)
        val_acc      = val_correct / val_total

        # — Log everything to W&B —
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss_avg,
            "val_acc":    val_acc,
        })

    run.finish()

# ─── Then your sweep invocation stays the same ───
sweep_cfg = {
    "method":"bayes",
    "metric": {"name":"val_acc","goal":"maximize"},
    "parameters":{
        "emb_dim":   {"values":[32,64,128]},
        "hid_dim":   {"values":[64,128]},
        "dropout":   {"values":[0.2,0.3]},
        "lr":        {"values":[1e-3,5e-4]},
        "epochs":   {"value":10}
    }
}

sweep_id = wandb.sweep(sweep_cfg, project="CS24M046_DA6401_A3_t1")
wandb.agent(sweep_id, train_attn_model, count=100)
