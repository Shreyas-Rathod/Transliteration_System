
import numpy as np
import pandas as pd
import json
import os

# ─── Configuration ────────────────────────────────────────────

# CSV with your test predictions:
# must have columns: input (source string), target (true target), pred (model prediction)
PRED_CSV = "predictions_attention.csv"

# NumPy file with attention weights:
# shape = (N_examples, T_output, S_input)
# must correspond in order to rows of PRED_CSV
ATTN_NPY = "attention_weights.npy"

# Output HTML
OUT_HTML = "attention_viz.html"


# ─── Load Data ────────────────────────────────────────────────

if not os.path.isfile(PRED_CSV):
    raise FileNotFoundError(f"Could not find {PRED_CSV}")
if not os.path.isfile(ATTN_NPY):
    raise FileNotFoundError(f"Could not find {ATTN_NPY}")

# Read predictions
df = pd.read_csv(PRED_CSV)
# Ensure columns exist
for col in ("input","target","pred"):
    if col not in df.columns:
        raise ValueError(f"{PRED_CSV} must contain column '{col}'")

input_texts  = df["input"].astype(str).tolist()
target_texts = df["target"].astype(str).tolist()
pred_texts   = df["pred"].astype(str).tolist()

# Load attention weights
attn = np.load(ATTN_NPY)  # shape (N, T, S)

N = len(input_texts)
if attn.shape[0] != N:
    raise ValueError(f"attention_weights.npy has {attn.shape[0]} examples but {PRED_CSV} has {N}")

# ─── Build HTML ───────────────────────────────────────────────

html = []
html.append("<!DOCTYPE html>")
html.append("<html><head><meta charset='utf-8'><title>Attention Connectivity</title>")
html.append("<style>")
html.append(" body { font-family: sans-serif; padding: 20px; }")
html.append(" .sample { margin-bottom: 2em; }")
html.append(" .charspan { font-family: monospace; display:inline-block; padding:4px; }")
html.append(" .output .charspan:hover { background:#eee; text-decoration:underline; cursor:pointer; }")
html.append("</style>")
html.append("</head><body>")
html.append("<h1>Seq2Seq Attention Connectivity</h1>")
html.append("<p>Hover over an <strong>output</strong> character to see which <strong>input</strong> chars it attended to.</p>")

for i in range(N):
    src = input_texts[i]
    tgt = pred_texts[i]   # visualize model’s prediction
    W   = attn[i]         # shape (T, S)

    html.append(f"<div class='sample' id='sample{i}'>")
    html.append(f"<h2>Sample {i+1}</h2>")
    html.append("<div><strong>Input:</strong> ")
    html.append("<span class='input'>")
    for c in src:
        html.append(f"<span class='charspan'>{c}</span>")
    html.append("</span></div>")

    html.append("<div><strong>Output:</strong> ")
    html.append("<span class='output'>")
    for c in tgt:
        html.append(f"<span class='charspan'>{c}</span>")
    html.append("</span></div>")

    # Embed the attention weights for this sample
    html.append("<script>")
    html.append(f"  (function(){{")
    html.append(f"    var W = {json.dumps(W.tolist())};")
    html.append(f"    var inp = document.querySelectorAll('#sample{i} .input .charspan');")
    html.append(f"    var out = document.querySelectorAll('#sample{i} .output .charspan');")
    html.append(f"    out.forEach((span, t) => {{")
    html.append(f"      span.addEventListener('mouseover', () => {{")
    html.append(f"        var row = W[t];")
    html.append(f"        var maxw = Math.max(...row);")
    html.append(f"        row.forEach((w, s) => {{")
    html.append(f"          var alpha = (maxw>0 ? w/maxw : 0);")
    html.append(f"          inp[s].style.backgroundColor = 'rgba(0,255,0,' + alpha.toFixed(3) + ')';")
    html.append(f"          inp[s].title = w.toFixed(3);")
    html.append(f"        }});")
    html.append(f"      }});")
    html.append(f"      span.addEventListener('mouseout', () => {{")
    html.append(f"        inp.forEach(el => {{ el.style.backgroundColor=''; el.title=''; }});")
    html.append(f"      }});")
    html.append(f"    }});")
    html.append(f"  }})();")
    html.append("</script>")

    html.append("</div>")  # end sample

html.append("</body></html>")

# Write to file
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print(f"✅ Attention visualization written to {OUT_HTML}")
