import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from torch.utils.data import DataLoader

# ── 5(b): Train on train+dev, then evaluate on test & log to W&B ──

# 1) Start W&B run
wandb.init(project="CS24M046_DA6401_A3_t1", name="attention_test_eval", reinit=True)

# 2) Paths & device
TRAIN = "/kaggle/input/gu-dataset/gu.translit.sampled.train.tsv"
DEV   = "/kaggle/input/gu-dataset/gu.translit.sampled.dev.tsv"
TEST  = "/kaggle/input/gu-dataset/gu.translit.sampled.test.tsv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Load data
def read_data(path):
    df = pd.read_csv(path, sep="\t", header=None).iloc[:, :2].dropna()
    return [(str(a), str(b)) for a,b in zip(df[0], df[1])]

train_data = read_data(TRAIN)
dev_data   = read_data(DEV)
test_data  = read_data(TEST)

# 4) Vocab & DataLoaders (reuse your functions & classes)
src_vocab, tgt_vocab = build_vocab(train_data + dev_data)
inv_tgt_vocab = {i:c for c,i in tgt_vocab.items()}

train_loader = DataLoader(TransliterationDataset(train_data+dev_data, src_vocab, tgt_vocab),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(TransliterationDataset(test_data,       src_vocab, tgt_vocab),
                          batch_size=32, shuffle=False, collate_fn=collate_fn)

# 5) Build model with your best hyperparams
best_cfg = {"emb_dim":64, "hid_dim":128, "dropout":0.2}
model = Seq2SeqAtt(
    emb_dim=best_cfg["emb_dim"],
    hid_dim=best_cfg["hid_dim"],
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    dropout=best_cfg["dropout"]
).to(device)

# 6) Retrain on train+dev for N epochs
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)
epochs = 5

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        outputs, _ = model(src, tgt)
        loss = criterion(outputs[:,1:].reshape(-1, outputs.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({"retrain_epoch": epoch, "retrain_loss": running_loss/len(train_loader)})

# 7) Evaluate on test set and build a W&B Table
model.eval()
correct = total = 0
table = wandb.Table(columns=["input","target","prediction"])
idx = 0

with torch.no_grad():
    for src_batch, tgt_batch in test_loader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        outputs, _ = model(src_batch, tgt_batch)
        preds = outputs.argmax(-1).cpu().tolist()
        truths = tgt_batch[:,1:].cpu().tolist()

        for p_seq, t_seq in zip(preds, truths):
            inp_str, true_str = test_data[idx]
            pred_str = "".join(inv_tgt_vocab[i] for i in p_seq if i!=0)
            table.add_data(inp_str, true_str, pred_str)
            if pred_str == true_str:
                correct += 1
            total += 1
            idx += 1

test_acc = correct/total * 100

# 8) Log results to W&B
wandb.log({
    "test_accuracy": test_acc,
    "test_predictions": table
})

print(f"Test accuracy: {test_acc:.2f}% — logged to W&B.")

