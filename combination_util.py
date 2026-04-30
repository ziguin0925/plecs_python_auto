import itertools
import numpy as np


# TODO : 파라미터 조합으로 메모리 많으면 DB나 yield 사용으로 변환
def generate_combination(
    parameter_dict: dict, chunk_size: int = None, use_modelvars: bool = True
) -> list:
    """
    :param parameter_dict: 입력 파라미터
    :param chunk_size: n으로나눌 단위 (None이면 전체 반환)
    :param use_modelvars: True → {'ModelVars': {...}} 형태
    :return [{},{},...]
    """

    keys = list(parameter_dict.keys())
    values = list(parameter_dict.values())

    combinations = []

    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        if use_modelvars:
            combinations.append({"ModelVars": param_dict})
        else:
            combinations.append(param_dict)

    # chunk 안 나누는 경우
    if chunk_size is None or chunk_size <= 0:
        return combinations

    # chunk로 나누기
    chunks = [
        combinations[i : i + chunk_size]
        for i in range(0, len(combinations), chunk_size)
    ]

    return chunks


def generate_combination_ESR(
    parameter_dict: dict, chunk_size: int = None, use_modelvars: bool = True
) -> list:

    keys = list(parameter_dict.keys())
    values = list(parameter_dict.values())

    combinations = []
    d = {
        "L1": (400e-6, 1300e-6),
        "L2": (10e-6, 190e-6),
        "C1": (1e-6, 10e-6),
        "C2": (10e-6, 190e-6),
    }

    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))

        param_dict["ESR_L1"] = round(param_dict["L1"] * 1000, 6)  # L1 ESR  0.4 ~ 1.3옴

        param_dict["ESR_L2"] = round(
            param_dict["L2"] * 1000, 6
        )  # L2 ESR  0.01 ~ 0.19옴
        # Rubycon 450V, 450PK1MEFC6.3X11
        param_dict["ESR_C1"] = round(
            0.25 / (2 * 3.14 * param_dict["C1"] * 50000), 6
        )  # Cpp ESR
        # ZLJ Rubycon 100V, 16ZLJ470MTA8X11.5
        param_dict["ESR_C2"] = round(
            0.08 / (2 * 3.14 * param_dict["C2"] * 50000), 6
        )  # 출력 커패시터 ESR

        if use_modelvars:
            combinations.append({"ModelVars": param_dict})
        else:
            combinations.append(param_dict)

    # chunk 안 나누는 경우
    if chunk_size is None or chunk_size <= 0:
        return combinations

    # chunk로 나누기
    chunks = [
        combinations[i : i + chunk_size]
        for i in range(0, len(combinations), chunk_size)
    ]

    return chunks


def generate_features(X):
    epsilon = 1e-8

    L1 = X[:, 0]
    L2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]

    features = [X]  # 원본 포함

    # -------------------
    # 1. Reciprocal (1/x)
    # -------------------
    recip = 1 / (X + epsilon)
    features.append(recip)

    # -------------------
    # 2. Pairwise product (x_i * x_j)
    # -------------------
    pair_products = []
    for i, j in itertools.combinations(range(4), 2):
        pair_products.append(X[:, i] * X[:, j])
        features.append(np.column_stack(pair_products))

    # -------------------
    # 3. Pairwise ratio (x_i / x_j)
    # -------------------
    pair_ratios = []
    for i, j in itertools.combinations(range(4), 2):
        pair_ratios.append(X[:, i] / (X[:, j] + epsilon))
        pair_ratios.append(X[:, j] / (X[:, i] + epsilon))
        features.append(np.column_stack(pair_ratios))

    # -------------------
    # 4. Inverse of products (1 / (x_i * x_j))
    # -------------------
    inv_products = []
    for i, j in itertools.combinations(range(4), 2):
        inv_products.append(1 / (X[:, i] * X[:, j] + epsilon))
        features.append(np.column_stack(inv_products))

    # -------------------
    # 최종 결합
    # -------------------
    return np.hstack(features)
