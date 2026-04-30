import pandas as pd
import numpy as np

# CSV 로드
df = pd.read_csv(
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/ripple_results.csv"
)


col = df["mosfet_cond_loss"]

min_idx = col.idxmin()
max_idx = col.idxmax()

print("min 값:", col[min_idx], "행 index:", min_idx)
print("max 값:", col[max_idx], "행 index:", max_idx)

print("mean", np.mean(col))
print("max", np.max(col))
print("min", np.min(col))
print("std", np.std(col))
print("var", np.var(col))


mask = df["mosfet_cond_loss"] > 400

filtered = df.loc[mask, ["L1", "L2", "C1", "C2", "mosfet_cond_loss"]]

print(len(filtered))
print(filtered[40:60])
