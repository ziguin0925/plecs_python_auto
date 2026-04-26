import os
import xmlrpc.client as xml
import numpy as np 
import combination_util as cu
from logger_config import get_logger
import data_save_util as dsu
import pandas as pd 

import time



call_back = """
function out_struct = compute_ripple(name, data, include_mode)

    if nargin < 3
        include_mode = true;
    end

    name = char(name); 

    data = data(:);
    start_idx = floor(length(data) * 0.7);
    data = data(start_idx:end);

    data_avg = mean(data);
    delta_i = max(data) - min(data);
    ripple_rate = delta_i / data_avg;

    if any(data < 0)
        mode = 'DCM';
    else
        mode = 'CCM';
    end

    out_struct = struct();
    out_struct.([name '_avg']) = data_avg;
    out_struct.(['delta_' name]) = delta_i;
    out_struct.([name '_ripple_rate']) = ripple_rate;

    if include_mode
        out_struct.([name '_Conduction_Mode']) = mode;
    end
end

slice = floor(length(result.Time) * 0.7);

L1 = result.Values(1,:);
L1_struct = compute_ripple("i_L1", L1);

L2 = result.Values(2,:);
L2_struct = compute_ripple("i_L2", L2);


Vout = result.Values(3,:);
Vout_struct = compute_ripple("V_out", Vout, false);


C1 = result.Values(4,:);
C1_struct = compute_ripple("v_C1", C1, false);

mosfet_cond_loss = mean(result.Values(5, slice:end));
mosfet_switch_loss = mean(result.Values(6, slice:end));
diode_cond_loss = mean(result.Values(7, slice:end));


L1_ESR_loss = mean(result.Values(8, slice:end));
L2_ESR_loss = mean(result.Values(9, slice:end));
C1_ESR_loss = mean(result.Values(10, slice:end));
C2_ESR_loss = mean(result.Values(11, slice:end));


leakage_rms = sqrt(mean(result.Values(12, :).^2));
efficiency = max(result.Values(13, :));



loss_struct = struct( ...
            "mosfet_cond_loss", mosfet_cond_loss, ...
            "mosfet_switch_loss", mosfet_switch_loss, ...
            "diode_cond_loss", diode_cond_loss, ...
            "L1_ESR_loss", L1_ESR_loss, ...
            "L2_ESR_loss", L2_ESR_loss, ...
            "C1_ESR_loss", C1_ESR_loss, ...
            "C2_ESR_loss", C2_ESR_loss, ...
            "leakage_rms", leakage_rms, ...
            "efficiency", efficiency ...
        );

out = struct(); 

fields = fieldnames(L1_struct);
for i = 1:numel(fields)
    out.(fields{i}) = L1_struct.(fields{i});
end

fields = fieldnames(L2_struct);
for i = 1:numel(fields)
    out.(fields{i}) = L2_struct.(fields{i});
end

fields = fieldnames(C1_struct);
for i = 1:numel(fields)
    out.(fields{i}) = C1_struct.(fields{i});
end


fields = fieldnames(Vout_struct);
for i = 1:numel(fields)
    out.(fields{i}) = Vout_struct.(fields{i});
end

fields = fieldnames(loss_struct);
for i = 1:numel(fields)
    out.(fields{i}) = loss_struct.(fields{i});
end

result = out;
"""

logger = get_logger()

param_specs = {
    "L1": {"start": 400e-6, "step": 100e-6, "count":10},
    "L2": {"start": 10e-6, "step": 20e-6, "count": 10},
    "C1": {"start": 1e-6, "step": 1e-6, "count": 10},
    "C2": {"start": 10e-6, "step": 20e-6, "count": 10},
}


parameter_dict = {
    key: [round(spec["start"] + spec["step"]*i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH ="./out/sepic_data/inner_loop"


def simulate_recursive(opts, plecs, name):
    if not opts:
        return []

    try:
        start = time.time()
        logger.info(f"[SIM START] [{opts}]")
        result = plecs.simulate(name, opts, call_back)
        logger.info(f"[SIM DONE] [SIZE = {len(opts)}] [elapsed = {time.time()-start:.2f}s]")
        return result

    except Exception as e:
        logger.info(f"[SIM FAIL] [{opts}]")
        if len(opts) == 1: # 1개 일 때는 그냥 실행
            logger.error(f"[PARAM ERROR] [PARAM = {opts}] [REASON = {e}]")
            return []

        mid = len(opts) // 2
        left = simulate_recursive(opts[:mid], plecs, name)
        right = simulate_recursive(opts[mid:], plecs, name)

        return left + right

def run_simulation(opts, plecs, name):
    result = simulate_recursive(opts, plecs, name)

    for idx, row in enumerate(result):
        current_parameters = opts[idx]["ModelVars"]

        for k, v in row.items():
            if isinstance(v, list):
                row[k] = v[0][0]

        result[idx] = {**current_parameters, **row}

    return result


MODEL_PATHS = [
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic1.plecs",
]
MODEL_NAMES = [os.path.splitext(os.path.basename(path))[0] for path in MODEL_PATHS]

# opts_lists = cu.generate_combination_ESR(parameter_dict, 15)

import existing_remainig as a
opts_lists = a.not_exist_parameter_opts()
logger.info(f"[opts_lists = {len(opts_lists)}]")
logger.info(f"[opts_lists[0] = {len(opts_lists[1])}]")


name = "sepic1"
plecs = xml.Server("http://localhost:1080/RPC2").plecs

if __name__ == "__main__":
    plecs.load(r"/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic1.plecs")
    
    
    for idx, opts in enumerate(opts_lists):
        logger.info(f"[SIM OPTS SEQUENCE = {idx+1}] [len(opts) = {len(opts)}]")
        save_data = run_simulation(opts=opts, plecs=plecs, name=name)
        if save_data:
            logger.info(f"[SAVE] : {opts}")
            dsu.data_to_csv(save_data, OUTPUT_PATH)






