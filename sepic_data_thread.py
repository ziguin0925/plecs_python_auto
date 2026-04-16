import pandas as pd
import numpy as np

file = pd.read_csv(r"/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/L2_sin.csv")

# 1. ripple_rate 계산
file["ripple_rate"] = (
    file["i_L1_ripple_rate"] +
    file["v_ripple_rate"] +
    np.abs(file["leakage_avg"])
)

# 2. CCM 조건 필터링
filtered = file[
    (file["Conduction_Mode_L1"] == "CCM") &
    (file["Conduction_Mode_L2"] == "CCM")
]

# 3. 가장 작은 순으로 상위 10개
top10 = filtered.nsmallest(10, "ripple_rate")

# 4. 원하는 컬럼 출력
result = top10[["L1", "L2", "C1", "F", "ripple_rate",
                "Conduction_Mode_L1", "Conduction_Mode_L2"]]

print(result)