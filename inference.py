import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure rnafm is in the path
sys.path.append("/mnt/")
# from rnafm.fm import pretrained as rnapretrained
from my_model import ProtRNA

# -----------------------------
# Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV with sgRNA_seq and off_seq columns")
parser.add_argument("--output_dir", required=True, help="Output CSV path")
args = parser.parse_args()

# -----------------------------
# Device
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# -----------------------------
# Load model
# -----------------------------
model = ProtRNA()
model.to(device)
model.eval()

# -----------------------------
# Get alphabet
# -----------------------------
alphabet_obj = model.get_alphabet()

# rnafm alphabets often have .get_vocab() or can cast to string
if hasattr(alphabet_obj, "get_vocab"):
    chars = alphabet_obj.get_vocab()
else:
    # fallback: try to convert to list
    chars = list(str(alphabet_obj))

char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding

# -----------------------------
# Tokenizer
# -----------------------------
def batch_tokenize(sequences, char2idx, device):
    max_len = max(len(seq) for seq in sequences)
    arr = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        arr[i, :len(seq)] = [char2idx.get(c, 0) for c in seq]
    return torch.tensor(arr, device=device)

# -----------------------------
# Load input CSV
# -----------------------------
df = pd.read_csv(args.input)
if "sgRNA_seq" not in df.columns or "off_seq" not in df.columns:
    raise ValueError("CSV must have columns: sgRNA_seq and off_seq")

pairs = [s + o for s, o in zip(df["sgRNA_seq"], df["off_seq"])]

# -----------------------------
# Batched inference
# -----------------------------
all_preds = []
B = 1024

with torch.no_grad():
    for i in tqdm(range(0, len(pairs), B), desc="Predicting"):
        batch = pairs[i:i + B]
        tokens = batch_tokenize(batch, char2idx, device=device)
        preds = model(tokens).to("cpu").numpy().flatten()
        all_preds.extend(preds)

# -----------------------------
# Save results
# -----------------------------
df["pred"] = all_preds
df.to_csv(args.output, sep="\t", index=False)
print(f"[SUCCESS] Predictions saved → {args.output}")
