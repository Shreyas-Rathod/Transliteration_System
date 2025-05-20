# Full Working Code for Question 2

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader

# Fix seeds
torch.manual_seed(42)
random.seed(42)

# WandB login
wandb.login(key='b4b7e96266129ba04e848c1f16f53dbf520f6c16')

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

# Dataset paths
TRAIN_PATH = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV_PATH = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"

def read_data(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.iloc[:, :2]  # Keep only the first two columns
    df = df.dropna()     # Drop rows with missing values
    return [(str(src), str(tgt)) for src, tgt in zip(df[0], df[1])]


train_data = read_data(TRAIN_PATH)
dev_data = read_data(DEV_PATH)

# Build vocabulary
def build_vocab(data):
    src_chars = set()
    tgt_chars = set()
    for src, tgt in data:
        src_chars.update(src)
        tgt_chars.update(tgt)
    src_vocab = {c: i+2 for i, c in enumerate(sorted(src_chars))}
    tgt_vocab = {c: i+2 for i, c in enumerate(sorted(tgt_chars))}
    src_vocab['<pad>'] = 0
    src_vocab['<sos>'] = 1
    tgt_vocab['<pad>'] = 0
    tgt_vocab['<sos>'] = 1
    return src_vocab, tgt_vocab

src_vocab, tgt_vocab = build_vocab(train_data + dev_data)
inv_tgt_vocab = {i: c for c, i in tgt_vocab.items()}

# Dataset class
class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = [self.src_vocab[c] for c in src]
        tgt_ids = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[c] for c in tgt]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

train_loader = DataLoader(TransliterationDataset(train_data, src_vocab, tgt_vocab), batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TransliterationDataset(dev_data, src_vocab, tgt_vocab), batch_size=32, collate_fn=collate_fn)

# Model
class Seq2Seq(nn.Module):
    def __init__(self, config, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.embedding_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.cell_type = config["cell_type"]

        self.encoder_embedding = nn.Embedding(src_vocab_size, self.embedding_dim, padding_idx=0)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, self.embedding_dim, padding_idx=0)

        rnn_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]

        self.encoder = rnn_class(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.decoder = rnn_class(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(self.hidden_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.dropout(self.encoder_embedding(src))
        encoder_outputs, hidden = self.encoder(embedded_src)

        if isinstance(hidden, tuple):  # LSTM
            hidden = tuple([self._adjust_layers(h) for h in hidden])
        else:
            hidden = self._adjust_layers(hidden)

        embedded_tgt = self.dropout(self.decoder_embedding(tgt))
        output, _ = self.decoder(embedded_tgt, hidden)
        output = self.fc(output)
        return output

    def _adjust_layers(self, h):
        if h.size(0) != self.num_layers:
            h = h.repeat(self.num_layers, 1, 1)
        return h

# Training function
def train_model():
    with wandb.init() as run:
        config = wandb.config
        model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to("cuda")
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(1, 6):
            # ---------- Training ----------
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for src, tgt in train_loader:
                src, tgt = src.to("cuda"), tgt.to("cuda")
                optimizer.zero_grad()
                output = model(src, tgt[:, :-1])
                loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(-1)
                mask = tgt[:, 1:] != 0
                correct += ((pred == tgt[:, 1:]) * mask).sum().item()
                total += mask.sum().item()

            train_acc = correct / total
            train_loss = total_loss / len(train_loader)

            # ---------- Validation ----------
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for src, tgt in dev_loader:
                    src, tgt = src.to("cuda"), tgt.to("cuda")
                    output = model(src, tgt[:, :-1])
                    loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
                    val_loss += loss.item()
                    pred = output.argmax(-1)
                    mask = tgt[:, 1:] != 0
                    val_correct += ((pred == tgt[:, 1:]) * mask).sum().item()
                    val_total += mask.sum().item()

            val_acc = val_correct / val_total
            val_loss_avg = val_loss / len(dev_loader)

            # ---------- Log ----------
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss_avg,
                "val_acc": val_acc,
            })


# Sweep config
sweep_config = {
    "method": "random",
    "metric": {"name": "train_acc", "goal": "maximize"},
    "parameters": {
        "embed_dim": {"values": [32, 64]},
        "hidden_size": {"values": [32, 64]},
        "num_layers": {"values": [1, 2]},
        "dropout": {"values": [0.2, 0.3]},
        "cell_type": {"values": ["RNN", "GRU", "LSTM"]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="CS24M046_DA6401_A3_t1")
wandb.agent(sweep_id, train_model, count=100)
