# ─── Question 4: Test‐time Evaluation (teacher‐forcing, W&B‐logged) ───

import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import wandb
from torch.utils.data import DataLoader

# 1) Start a new W&B run
wandb.init(
    project="CS24M046_DA6401_A3_t1",
    name="test_evaluation",
    reinit=True
)

# 2) File paths
TRAIN = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV   = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"
TEST  = "/kaggle/input/gu-dataset/gu.translit.sampled.test.tsv"

# 3) Re‐use your read_data, build_vocab, Dataset, collate_fn, Seq2Seq
def read_data(path):
    df = pd.read_csv(path, sep="\t", header=None).iloc[:, :2].dropna()
    return [(str(a), str(b)) for a,b in zip(df[0], df[1])]

train_data = read_data(TRAIN)
dev_data   = read_data(DEV)
test_data  = read_data(TEST)

src_vocab, tgt_vocab = build_vocab(train_data+dev_data)
inv_tgt_vocab = {i:c for c,i in tgt_vocab.items()}

train_loader = DataLoader(TransliterationDataset(train_data+dev_data, src_vocab, tgt_vocab),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(TransliterationDataset(test_data,       src_vocab, tgt_vocab),
                          batch_size=32, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_cfg = {
    "embed_dim":   64,
    "hidden_size": 64,
    "num_layers":  2,
    "dropout":     0.3,
    "cell_type":  "LSTM"
}
model = Seq2Seq(best_cfg, len(src_vocab), len(tgt_vocab)).to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.CrossEntropyLoss(ignore_index=0)

# 4) (Re)train on train+dev for 5 epochs
for epoch in range(1,6):
    model.train()
    running = 0.0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out = model(src, tgt[:, :-1])  
        loss = crit(out.view(-1,out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward(); opt.step()
        running += loss.item()
    wandb.log({"retrain_epoch": epoch, "retrain_loss": running/len(train_loader)})

# 5) Evaluate on test set via teacher‐forcing
model.eval()
total = correct = 0
table = wandb.Table(columns=["input","target","prediction"])

with torch.no_grad():
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        out = model(src, tgt[:, :-1])              # [B, T-1, V]
        preds = out.argmax(-1).cpu().tolist()      # [B, T-1]
        truths = tgt[:,1:].cpu().tolist()          # [B, T-1]

        for (pseq,tseq), (inp_str, _) in zip(zip(preds, truths), test_data[0:len(preds)]):
            # reconstruct strings
            p_str = "".join(inv_tgt_vocab[i] for i in pseq if i!=0)
            t_str = "".join(inv_tgt_vocab[i] for i in tseq if i!=0)
            table.add_data(inp_str, t_str, p_str)
            correct += (p_str == t_str)
            total   += 1

test_acc = 100 * correct/total
wandb.log({"test_accuracy": test_acc, "test_predictions": table})
print(f"Test exact‐match accuracy: {test_acc:.2f}% logged to W&B.")
