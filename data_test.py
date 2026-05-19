import data_analysis.raw_data_visualize as rdv
import pandas as pd

name_list = [
    "C1_ESR_loss",
    "C2_ESR_loss",
    "L1_ESR_loss",
    "L2_ESR_loss",
    "diode_cond_loss",
    "mosfet_cond_loss",
    "mosfet_switch_loss",
    "i_L1_ripple_rate",
    # "i_L2_ripple_rate",
    # "v_C1_ripple_rate",
    "V_out_ripple_rate",
    "total_efficiency",
]

df = pd.read_csv(
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/result_file_sepic_no_under_cpp.csv"
)

print(len(df))  # 9897
df = df[
    (df["mosfet_cond_loss"] < 1000)
    & (df["i_L1_ripple_rate"] <= 0.2)
    & (df["V_out_ripple_rate"] <= 0.02)
]
print(len(df))  # 8994

print(df[name_list].describe())

rdv.data_distribution_plot(df, name_list, plot_type="box")


x_list = ["L1", "L2", "C1", "fs"]
y_ESR_list = ["L1_ESR_loss", "L2_ESR_loss", "C1_ESR_loss", "C2_ESR_loss"]
y_mos_dio_list = ["diode_cond_loss", "mosfet_cond_loss", "mosfet_switch_loss"]
y_ripple_list = ["i_L1_ripple_rate", "V_out_ripple_rate"]
y_efficiency_list = ["total_efficiency"]

rdv.seaborn_pairplot(df, x_list, y_ESR_list)
rdv.seaborn_pairplot(df, x_list, y_mos_dio_list)
rdv.seaborn_pairplot(df, x_list, y_ripple_list)
rdv.seaborn_pairplot(df, x_list, y_efficiency_list)
