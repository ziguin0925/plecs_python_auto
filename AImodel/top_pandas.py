import pandas as pd

# CSV 로드
df = pd.read_csv("experiment_results.csv")

# best_val_accuracy 기준 내림차순 정렬 후 상위 5개
top5 = df.sort_values(by="best_val_accuracy", ascending=False).head(10)

# 보고 싶은 컬럼만 선택
cols = [
    "batch_size",
    "epochs",
    "learning_rate",
    "num_conv_blocks",
    "filters",
    "kernel_size",
    "use_batchnorm",
    "dropout_rate",
    "dense_units",
    "best_val_accuracy"
]

print(top5[cols])