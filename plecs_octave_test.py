import os
import xmlrpc.client as xml
import numpy as np
import combination_util as cu
from logger_config import get_logger
import data_save_util as dsu

import time

call_back = open(
    r"/home/pcsl/Documents/plecs/sepic/plecs_python_auto/octave_file/octave.m"
).read()
L1 = 0.000900
L2 = 0.00017
C1 = 12e-06
C2 = 1e-05


logger = get_logger()
param_specs = {
    "L1": {"start": 900e-6, "step": 100e-6, "count": 1},
    "L2": {"start": 170e-6, "step": 50e-6, "count": 1},
    "C1": {"start": 12e-6, "step": 100e-6, "count": 1},
    "C2": {"start": 10e-6, "step": 5e3, "count": 1},
}


parameter_dict = {
    key: [round(spec["start"] + spec["step"] * i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
}

OUTPUT_PATH = "./out/sepic_data"


def run_simulation(opts, plecs, name):
    try:
        start = time.time()
        result = plecs.simulate(name, opts, call_back)
        logger.info(f"[SIM DONE] : {opts},     elapsed={time.time()-start:.2f}s")
    except Exception as e:
        logger.error(
            f"[Error in simulate: {e}] \n {opts}"
        )  # In 260313 AC_DC_sin_PFC_JH4/Modulator/Divide: Division by zero. 이거 많이남
        return []

    logger.info(f"result = [{result}]")
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
            dsu.data_to_csv(save_data, OUTPUT_PATH, "result_file")
