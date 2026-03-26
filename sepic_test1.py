import xmlrpc.client as xml
import numpy as np 
import combination_util as cu
import data_save_util as dsu
from logger_config import get_logger
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


logger = get_logger()

param_specs = {
    "L1": {"start": 300e-6, "step": 50e-6, "count": 3},
    "L2": {"start": 300e-6, "step": 30e-6, "count": 4},
}

parameter_dict = {
    key: [round(spec["start"] + spec["step"]*i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH ="./out/sepic_data"



def compute_ripple(name:str, data:list, include_mode: bool = True) -> dict:
    data = np.asarray(data)[int(len(data) * 0.8):] # 뒤 20% 구간 슬라이싱함
    data_avg = float(data.mean())
    delta_i_data = float(data.max() - data.min())
    data_ripple_rate = delta_i_data / data_avg
    conduction_mode_data = "DCM" if (data < 0).any() else "CCM"

    result_dict = {
        f"i_{name}_avg": data_avg,
        f"delta_i_{name}": delta_i_data,
        f"{name}_ripple_rate": data_ripple_rate,
    }

    if include_mode:
        result_dict[f"{name}_Conduction_Mode"] = conduction_mode_data

    return result_dict

def run_simulation(opts):
    try:
        result = plecs.simulate(model, opts)
    except Exception as e:
        logger.error(f"[Error in simulate: {e}]")
        return []

    all_save_data = []
    for idx, data in enumerate(result):
        current_parameters = opts[idx]["ModelVars"]

        L1 = data['Values'][1]
        L1_dict_data = compute_ripple("L1", L1)
        L2 = data['Values'][2]
        L2_dict_data = compute_ripple("L2", L2)

        save_data = {**current_parameters, **L1_dict_data, **L2_dict_data}
        all_save_data.append(save_data)

    return all_save_data


opts_lists = cu.generate_combination(parameter_dict, 3)

model = '260313 AC_DC_sin_PFC_JH4'
file_type = '.plecs'
plecs = xml.Server("http://localhost:1080/RPC2").plecs

if __name__ == "__main__":
    plecs.load(r"C:\Users\KNU007\Documents\python_prac\plecs\260313 AC_DC_sin_PFC_JH4.plecs")
    num_cores = 4 
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run_simulation, opts) for opts in opts_lists]

        for future in as_completed(futures):
            save_data = future.result()
            if save_data:
                dsu.data_to_csv(save_data, OUTPUT_PATH)

    # plecs.close(model)


