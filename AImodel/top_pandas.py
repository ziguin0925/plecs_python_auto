import pandas as pd
import numpy as np

# CSV 로드
df = pd.read_csv(
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/result_file_sepic_no_under_cpp.csv"
)
print(len(df))

# V_out_ripple_rate var : 0.19290540373195775
# min 값: 0.0120469085489941 행 index: 2502
# max 값: 3.51568424124514 행 index: 8852
# mean 0.42740372017055867
# max 3.51568424124514
# min 0.0120469085489941
# std 0.4392099768128654
# var 0.19290540373195775

# diode_cond_loss
# min 값: 51.73157458193327 행 index: 3488
# max 값: 69.77590387619578 행 index: 44
# mean 55.06531083975695
# max 69.77590387619578
# min 51.73157458193327
# std 3.8443708583993317
# var 14.779187296910015

# i_L1_ripple_rate
# min 값: 0.1183333160416765 행 index: 8795
# max 값: 15.039348090032153 행 index: 62
# mean 2.927856972095069
# max 15.039348090032153
# min 0.1183333160416765
# std 3.718452438750599
# var 13.826888539250277

# col = df["i_L1_ripple_rate"]

# min_idx = col.idxmin()
# max_idx = col.idxmax()

# print("min 값:", col[min_idx], "행 index:", min_idx)
# print("max 값:", col[max_idx], "행 index:", max_idx)

# print("mean", np.mean(col))
# print("max", np.max(col))
# print("min", np.min(col))
# print("std", np.std(col))
# print("var", np.var(col))


# mask = df["mosfet_cond_loss"] < 100

# filtered = df.loc[mask, ["L1", "L2", "C1", "C2", "mosfet_cond_loss"]]

# print(len(filtered))
# print(filtered[40:60])


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# cols = [
#     "i_L1_ripple_rate",
#     "diode_cond_loss",
#     "V_out_ripple_rate",
#     "mosfet_cond_loss",
#     "mosfet_switch_loss",
# ]

# data = df[cols].dropna()

# for c in cols:
#     plt.figure()

#     x = data[c].values

#     # histogram
#     plt.hist(x, bins=50, density=True, alpha=0.6)

#     # gaussian fit
#     mu, sigma = norm.fit(x)
#     xmin, xmax = x.min(), x.max()
#     xx = np.linspace(xmin, xmax, 200)
#     plt.plot(xx, norm.pdf(xx, mu, sigma), "r", linewidth=2)

#     plt.title(f"{c} (μ={mu:.3f}, σ={sigma:.3f})")
#     plt.xlabel("Value")
#     plt.ylabel("Density")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# path = (
#     "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/result_file.csv"
# )
# df = pd.read_csv(path)
# row = df.iloc[0]

# leakage_cols = sorted([c for c in df.columns if c.startswith("leakage_stamp_")])
# t = [float(c.replace("leakage_stamp_", "")) for c in leakage_cols]
# t = [x * 2 - 0.01 for x in t]
# y = [float(row[c]) for c in leakage_cols]

# # mA 단위로 변환 (A → mA)
# y_mA = [val * 1000 for val in y]

# plt.figure(figsize=(8, 5))
# plt.plot(y_mA, t, marker="o", linewidth=2, markersize=6)  # ← x, y 순서 swap
# plt.xlabel("Leakage (mA)")
# plt.ylabel("Time (s)")
# plt.title("Time vs Leakage (row 1)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
