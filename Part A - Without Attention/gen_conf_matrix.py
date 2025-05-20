import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 1) Load your CSV export
csv_path = "/kaggle/input/predictedtest/all_pred.csv"
df = pd.read_csv(csv_path).dropna(subset=['target','prediction'])

# 2) Build lists of aligned true vs. pred chars
y_true, y_pred = [], []
for _, row in df.iterrows():
    t, p = str(row['target']), str(row['prediction'])
    L = min(len(t), len(p))
    y_true.extend(list(t[:L]))
    y_pred.extend(list(p[:L]))

# 3) Define the label set
labels = sorted(set(y_true + y_pred))

# 4) Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 5) Plot with Matplotlib
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(cm, interpolation='nearest', aspect='auto')

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

ax.set_xlabel('Predicted Character')
ax.set_ylabel('True Character')
ax.set_title('Character-level Confusion Matrix')
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
