import pandas as pd

param_specs = {
    "L1": {"start": 400e-6, "step": 100e-6, "count":10},
    "L2": {"start": 10e-6, "step": 20e-6, "count": 10},
    "C1": {"start": 1e-6, "step": 1e-6, "count": 10},
    "C2": {"start": 10e-6, "step": 20e-6, "count": 10},
}


def not_exist_parameter_opts():
    df = pd.read_csv("/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/ripple_results.csv")

    existing_set = set(
        tuple(row)
        for row in df[["L1", "L2", "C1", "C2"]].values
        )
    parameter_dict = {
    key: [round(spec["start"] + spec["step"]*i, 6) for i in range(spec["count"])]
    for key, spec in param_specs.items()
    }

    import itertools

    all_combinations = []

    keys = list(parameter_dict.keys())
    values = list(parameter_dict.values())

    remaining = []

    for combo in itertools.product(*values):
        comb = dict(zip(keys, combo))
        key = (comb["L1"], comb["L2"], comb["C1"], comb["C2"])
        
        if key not in existing_set:
            comb["ESR_L1"] = round(comb["L1"] * 1000, 6)
            comb["ESR_L2"] = round(comb["L2"] * 1000, 6)
            comb["ESR_C1"] = round(0.25 / (2 * 3.14 * comb["C1"] * 50000), 6)
            comb["ESR_C2"] = round(0.08 / (2 * 3.14 * comb["C2"] * 50000), 6)

            remaining.append(comb)

    opts_lists = [{"ModelVars": r} for r in remaining]
    chunked = [opts_lists[i:i+10] for i in range(0, len(opts_lists), 10)]
    
    return chunked


a = not_exist_parameter_opts()
print(len(a))