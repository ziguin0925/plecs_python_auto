import os
import xmlrpc.client as xml
import numpy as np
import combination_util as cu
from logger_config import get_logger
import data_save_util as dsu
import pandas as pd

import time

logger = get_logger()

call_back = open(
    "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/AImodel/sepic_test/octave.m"
).read()

param_specs = {
    "L1": {"start": 400e-6, "step": 100e-6, "count": 10},
    "L2": {"start": 10e-6, "step": 20e-6, "count": 10},
    "C1": {"start": 1e-6, "step": 1e-6, "count": 10},
    "C2": {"start": 10e-6, "step": 20e-6, "count": 10},
}


parameter_dict = {
    key: [round(spec["start"] + spec["step"] * i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH = "./out/sepic_data/inner_loop"


def simulate_recursive(opts, plecs, name):
    if not opts:
        return []

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
            logger.error(
                f"[PARAM ERROR] [PARAM = {opts}] [REASON = {e}] [RESULT = {result}]"
            )
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
            dsu.data_to_csv(save_data, OUTPUT_PATH, "result_file")
