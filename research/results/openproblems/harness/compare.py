"""Ours vs published scores (run_2024-12-19) for one dataset."""
import sys
import pandas as pd

ours = pd.read_csv(sys.argv[1])
pub = pd.read_csv(sys.argv[2])
ds = ours.dataset.iloc[0]
p = pub[pub.dataset == ds].pivot_table(index="method", columns="metric", values="value")
cols = [c for c in p.columns if c in ours.columns]
o = ours.set_index("method")[cols]
o.index = [f"{m} (ours)" for m in o.index]
t = pd.concat([o, p[cols]]).T
pd.set_option("display.width", 300)
print(t.round(3).to_string())
