import xmlrpc.client as xml
import numpy as np 
import combination_util as cu
import data_save_util as dsu
from logger_config import get_logger

import time



logger = get_logger()

param_specs = {
    "L1": {"start": 100e-6, "step": 100e-6, "count": 10},
    "L2": {"start": 10e-6, "step": 50e-6, "count": 10},
    "C1": {"start": 10e-6, "step": 50e-6, "count": 10},
    "fs": {"start": 60e3, "step": 5e3, "count": 10},
}

# param_specs = {
#     "L1": {"start": 100e-6, "step": 100e-6, "count": 10},
#     "L2": {"start": 10e-6, "step": 50e-6, "count": 10},
#     "C1": {"start": 10e-6, "step": 50e-6, "count": 10},
#     "fs": {"start": 60e3, "step": 5e3, "count": 10}, #85일때 에러
# }


parameter_dict = {
    key: [round(spec["start"] + spec["step"]*i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH ="./out/sepic_data"



def compute_ripple(name:str, data:list, include_mode: bool = True) -> dict:
    data = np.asarray(data)[int(len(data) * 0.7):] # 뒤 20% 구간 슬라이싱함
    data_avg = float(data.mean())
    delta_i_data = float(data.max() - data.min())
    data_ripple_rate = delta_i_data / data_avg
    conduction_mode_data = "DCM" if (data < 0).any() else "CCM"

    result_dict = {
        f"{name}_avg": data_avg,
        f"delta_i_{name}": delta_i_data,
        f"{name}_ripple_rate": data_ripple_rate,
    }

    if include_mode:
        result_dict[f"{name}_Conduction_Mode"] = conduction_mode_data

    return result_dict

def run_simulation(opts, plecs, name):
    try:
        start = time.time()
        result = plecs.simulate(name, opts)
        logger.info(f"[SIM DONE] : {opts},     elapsed={time.time()-start:.2f}s")
    except Exception as e:
        logger.error(f"[Error in simulate: {e}] \n {opts}") # In 260313 AC_DC_sin_PFC_JH4/Modulator/Divide: Division by zero. 이거 많이남
        return []

    all_save_data = []
    for idx, data in enumerate(result):
        current_parameters = opts[idx]["ModelVars"]

        slice = int(len(data['Time'])*0.8)

        L1 = data['Values'][0]
        L1_dict_data = compute_ripple("i_L1", L1)

        L2 = data['Values'][1]
        L2_dict_data = compute_ripple("i_L2", L2)
        
        V_out = data['Values'][2]
        Vout_dict_data = compute_ripple("V_out", V_out, False)
        
        mosfet_temp = np.asarray(data['Values'][3][slice:]).mean()
        # diode_temp = data['Values'][4]

        mosfet_cond_loss = np.asarray(data['Values'][4][slice:]).mean()
        # diode_cond_loss = data['Values'][5]

        mosfet_switch_loss = np.asarray(data['Values'][5][slice:]).mean()
        # diode_switch_loss = data['Values'][6]

        mosfet_dict = {
            "mosfet_temp" : mosfet_temp,
            "mosfet_cond_loss" : mosfet_cond_loss,
            "mosfet_switch_loss" : mosfet_switch_loss,
        }

        save_data = {**current_parameters, **L1_dict_data, **L2_dict_data, **Vout_dict_data, **mosfet_dict}
        all_save_data.append(save_data)

    return all_save_data


opts_lists = cu.generate_combination(parameter_dict, 1)
logger.info(f"opts_lists = {len(opts_lists)}")
logger.info(f"opts_lists[0] = {len(opts_lists[0])}")

name = 'sepic'
file_type = '.plecs'
plecs = xml.Server("http://localhost:1080/RPC2").plecs

if __name__ == "__main__":
    plecs.load(r"/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic.plecs")
    num_cores = 1
    
    for opts in opts_lists:
        save_data = run_simulation(opts=opts, plecs=plecs, name=name)
        if save_data:
            logger.info(f"[SAVE] : {opts}")
            dsu.data_to_csv(save_data, OUTPUT_PATH)

    # plecs.close(model)


