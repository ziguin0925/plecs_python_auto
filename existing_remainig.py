import pandas as pd
import numpy as np

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


def not_exist_parameter_opts(parameter_dict):
    df = pd.read_csv(
        "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/result_file_sepic_no_under_cpp.csv"
    )

    existing_set = set(tuple(row) for row in df[["L1", "L2", "C1", "fs"]].values)

    import itertools

    all_combinations = []

    keys = list(parameter_dict.keys())
    values = list(parameter_dict.values())

    remaining = []

    for combo in itertools.product(*values):
        comb = dict(zip(keys, combo))
        key = (comb["L1"], comb["L2"], comb["C1"], comb["fs"])

        if key not in existing_set:
            comb["ESR_L1"] = round(comb["L1"] * 1000, 6)
            comb["ESR_L2"] = round(comb["L2"] * 1000, 6)
            comb["ESR_C1"] = round(0.25 / (2 * 3.14 * comb["C1"] * comb["fs"]), 6)
            # comb["ESR_C2"] = round(0.08 / (2 * 3.14 * 70e-06 * comb["fs"]), 6)

            remaining.append(comb)

    opts_lists = [{"ModelVars": r} for r in remaining]
    chunked = [opts_lists[i : i + 10] for i in range(0, len(opts_lists), 10)]

    return chunked
