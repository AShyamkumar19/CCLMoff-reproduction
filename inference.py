import torch 
import pandas as pd
from my_model import ProtRNA
from utils import batch_tokenize 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t")
sg = df["sgRNA_seq"].tolist()
off = df["off_seq"].tolist()

pairs = [s + o for s, o in zip(sg, off)]

model = ProtRNA()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

alphabet = model.get_alphabet()
tokens = batch_tokenize(pairs, alphabet, device=device)

with torch.no_grad():
    preds = model(tokens).cpu().numpy().flatten()

df["pred"] = preds
df.to_csv(args.output, sep="\t", index=False)

print(f"[SUCCESS] Predictions saved → {args.output}")
