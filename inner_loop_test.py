import os
from pathlib import Path
import xmlrpc.client as xml
import numpy as np
import combination_util as cu
from logger_config import get_logger
import data_save_util as dsu
import pandas as pd

import time

logger = get_logger()

call_back = open(
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/octave_file/no_cpp_octave.m"
).read()

param_specs = {
    "L1": {"min": 220e-6, "max": 1700e-6, "count": 10},
    "L2": {"min": 25e-6, "max": 205e-6, "count": 10},
    "C1": {"min": 1e-6, "max": 25e-6, "count": 10},
    "fs": {"min": 50e3, "max": 100e3, "count": 10, "round": -2},
}

parameter_dict = {
    key: (
        np.round(
            np.linspace(spec["min"], spec["max"], spec["count"]),
            spec.get("round", 6),
        ).tolist()
    )
    for key, spec in param_specs.items()
}

OUTPUT_PATH = "./out/sepic_data/inner_loop"


def simulate_recursive(opts, plecs, name):
    if not opts:
        return []

    result = None
    try:
        start = time.time()
        logger.info(f"[SIM START] [{opts}]")
        result = plecs.simulate(name, opts, call_back)
        logger.info(
            f"[SIM DONE] [SIZE = {len(opts)}] [elapsed = {time.time()-start:.2f}s]"
        )
        return result

    except Exception as e:
        logger.info(f"[SIM FAIL] [{opts}]")
        if len(opts) == 1:  # 1개 일 때는 그냥 실행
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


MODEL_PATH = (
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/sepic_no_under_cpp.plecs"
)

# opts_lists = cu.generate_combination_ESR(parameter_dict, 12)

import existing_remainig as a

opts_lists = a.not_exist_parameter_opts(parameter_dict)
logger.info(f"[opts_lists = {len(opts_lists)}]")
logger.info(f"[opts_lists[0] = {len(opts_lists[1])}]")
#

plecs = xml.Server("http://localhost:1080/RPC2").plecs

if __name__ == "__main__":
    plecs.load(MODEL_PATH)  # change here

    for idx, opts in enumerate(opts_lists):
        logger.info(f"[SIM OPTS SEQUENCE = {idx+1}] [len(opts) = {len(opts)}]")
        save_data = run_simulation(opts=opts, plecs=plecs, name=Path(MODEL_PATH).stem)
        if save_data:
            logger.info(f"[SAVE] : {opts}")
            dsu.data_to_csv(save_data, OUTPUT_PATH, "result_file_sepic_no_under_cpp")
