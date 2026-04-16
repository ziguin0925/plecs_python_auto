import os
import xmlrpc.client as xml
import numpy as np 
import combination_util as cu
from logger_config import get_logger
import data_save_util as dsu

import time



call_back = """
function result = compute_ripple(name, data, include_mode)

    if nargin < 3
        include_mode = true;
    end

    data = data(:);
    start_idx = floor(length(data) * 0.7);
    data = data(start_idx:end);

    data_avg = mean(data);
    delta_i = max(data) - min(data);
    ripple_rate = delta_i / data_avg;

    if any(data < 0)
        mode = "DCM";
    else
        mode = "CCM";
    end

    result = struct();
    result.([name '_avg']) = data_avg;
    result.(['delta_' name]) = delta_i;
    result.([name '_ripple_rate']) = ripple_rate;

    if include_mode
        result.([name '_Conduction_Mode']) = mode;
    end
end

slice = floor(length(result.Time) * 0.7);

L1 = result.Values(1,:);
L1_struct = compute_ripple("i_L1", L1);

L2 = result.Values(2,:);
L2_struct = compute_ripple("i_L2", L2);

Vout = result.Values(3,:);
Vout_struct = compute_ripple("V_out", Vout, false);

mosfet_cond_loss = mean(result.Values(4, slice:end));
mosfet_switch_loss = mean(result.Values(5, slice:end));

loss_struct = struct( ...
            "mosfet_cond_loss", mosfet_cond_loss, ...
            "mosfet_switch_loss", mosfet_switch_loss ...
        );

result=struct();

fields = fieldnames(L1_struct);
for i = 1:numel(fields)
    result.(fields{i}) = L1_struct.(fields{i});
end

fields = fieldnames(L2_struct);
for i = 1:numel(fields)
    result.(fields{i}) = L2_struct.(fields{i});
end

fields = fieldnames(Vout_struct);
for i = 1:numel(fields)
    result.(fields{i}) = Vout_struct.(fields{i});
end

fields = fieldnames(loss_struct);
for i = 1:numel(fields)
    result.(fields{i}) = loss_struct.(fields{i});
end
"""

logger = get_logger()

param_specs = {
    "L1": {"start": 100e-6, "step": 100e-6, "count":10},
    "L2": {"start": 10e-6, "step": 50e-6, "count": 10},
    "C1": {"start": 10e-6, "step": 50e-6, "count": 10},
    "fs": {"start": 60e3, "step": 5e3, "count": 10},
}


parameter_dict = {
    key: [round(spec["start"] + spec["step"]*i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH ="./out/sepic_data"


def run_simulation(opts, plecs, name):
    try:
        start = time.time()
        result = plecs.simulate(name, opts, call_back)
        logger.info(f"[SIM DONE] : {opts},     elapsed={time.time()-start:.2f}s")
    except Exception as e:
        logger.error(f"[Error in simulate: {e}] \n {opts}") # In 260313 AC_DC_sin_PFC_JH4/Modulator/Divide: Division by zero. 이거 많이남
        return []

    for idx, row in enumerate(result):
        current_parameters = opts[idx]["ModelVars"]
        for k, v in row.items():
            if isinstance(v, list):
                row[k] = v[0][0]  # 이중 리스트 -> 숫자
        result[idx] = {**current_parameters, **row}

    return result


MODEL_PATHS = [
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic1.plecs",
]
MODEL_NAMES = [os.path.splitext(os.path.basename(path))[0] for path in MODEL_PATHS]

opts_lists = cu.generate_combination(parameter_dict, 20)
logger.info(f"opts_lists = {len(opts_lists)}")
logger.info(f"opts_lists[0] = {len(opts_lists[0])}")

name = "sepic1"
plecs = xml.Server("http://localhost:1080/RPC2").plecs

if __name__ == "__main__":
    plecs.load(r"/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic1.plecs")
    
    
    for opts in opts_lists:
        save_data = run_simulation(opts=opts, plecs=plecs, name=name)
        if save_data:
            logger.info(f"[SAVE] : {opts}")
            dsu.data_to_csv(save_data, OUTPUT_PATH)




