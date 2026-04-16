import numpy as np
import math
import random
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path



random.seed(77)


def bounds() -> dict:
    '''
    탐색 범위 정의
    '''
    return{
        "fsw" : (6.000e+04 , 1.050e+05),
        "L1" : (1.000e-04, 1.000e-03),
        "L2" : (1.000e-05, 4.600e-04),
        "C" : (1.000e-05, 4.600e-04),
    }
def constraints() -> list:
    '''
    성능 제약 조건 정의
    '''
    return[
        ("lripple", 2.0, '<='),
        ("vripple", 20.0, "<="),
    ]

def evaluate_lt(model, scaler, x) -> dict:
  
	x_array = np.array([[x["L1"], x["L2"], x["C"], x["fsw"]]])
	x_scaled = scaler["std_x"].transform(x_array)
	
	y_ripple_scaled = model["ripple"].predict(x_scaled)
	y_loss_scaled  = model["loss"].predict(x_scaled, verbose=0)
	y_temp_scaled  = model["temp"].predict(x_scaled, verbose=0)
	
	y_ripple_pred = scaler["std_ripple"].inverse_transform(y_ripple_scaled)
	y_loss_pred = scaler["std_loss"].inverse_transform(y_loss_scaled)
	y_temp_pred = scaler["std_temp"].inverse_transform(y_temp_scaled)

    
	return {
		"lripple" : y_ripple_pred[0, 0],
		"vripple" : y_ripple_pred[0, 1],
		"cond_loss" : y_loss_pred[0, 0],
		"sw_loss" : y_loss_pred[0, 1],
		"temp" : y_temp_pred[0, 0],
	}

def clamp(x, lo, hi):
    '''
    lo ~ hi 값만 나오도록
    '''
    # 삼항 연산자
    return lo if x < lo else hi if x > hi else x

def random_solution(bnd):
    '''
     lo ~ hi사이의 실수 중에서 난수값 return
     :param bnd: fsw, L, C의 최소 최대값
    '''
    x = {}
    for k in bnd:
        lo, hi = bnd[k]
        x[k] = random.uniform(lo, hi) 
    return x

def objective(metrics, scaler, p=6):
    '''
    p-norm 기반 목적 함수 (minimize)
    '''

    ripple = np.array([[metrics["lripple"], metrics["vripple"]]])
    ripple_norm = scaler["mm_ripple"].transform(ripple) # minmax로 바꾸기(음수 피함)


    loss = metrics["cond_loss"] + metrics["sw_loss"]

    i = ripple_norm[0, 0]
    v = ripple_norm[0, 1]

    return loss + (v**p + i**p)**(1/p)

def penalty(metrics, rho=1e6):
    '''
    제약조건 위반 시 벌점 부여 - constraints에 적힌 것 안에서만
    '''
    total = 0.0
    
    for name, limit, sense in constraints():
        value = metrics[name]

        # 물리적으로 음수 방지 (ripple, leakage >= 0)
        if value < 0:
            total += 1000000 * (value ** 2)
        else:
            # 일반 제약조건 위반
            if sense == "<=":
                viol = max(0.0, value - limit)
            else:
                viol = max(0.0, limit - value)

            total += viol ** 2

    return rho * total

def harmony_search(
  		model,
  		scaler,
        hms = 30,
        hmcr = 0.95,
        par = 0.7,
        max_iters = 10,
        rho = 1e6,
):
    bnd = bounds()
    bw = {}

    # 각 f, L, C의 범위 차에 0.02를 곱해줌 - 미세조정을 위해서
    for k in bnd:
        lo, hi = bnd[k]
        bw[k] = 0.2 * (hi - lo)

    HM = []
    HM_fit = []
    HM_metrics = []

    print("Initialize HM ...")
    # 처음 F, L, C조합 30개에 대한 fit값 추출
    for _ in range(hms):
        x = random_solution(bnd)
        m = evaluate_lt(model, scaler, x)
        f = objective(m, scaler) + penalty(m, rho)
        HM.append(x)
        HM_metrics.append(m)
        HM_fit.append(f)

    # 적응도가 낮아야 좋음
    # 적응도를 오름차순으로 HM, HM_metrics, HM_fit 정렬
    idx = sorted(range(hms), key = lambda i : HM_fit[i])
    HM = [HM[i] for i in idx]
    HM_metrics = [HM_metrics[i] for i in idx]
    HM_fit = [HM_fit[i] for i in idx]

    best_x = HM[0]
    best_m = HM_metrics[0]
    best_f = HM_fit[0]

    history = [best_f]

    for it in range(max_iters):
        x_new = {}
        # x_new를 새로운 값으로 초기화
        for k in bnd:
            lo, hi = bnd[k]
            if random.random() < hmcr:
                sel = random. randint(0, hms - 1) # 0 ~ 29
                val = HM[sel][k]
                if random.random() < par: # 미세조정
                    val += random.uniform(-bw[k], bw[k])
                x_new [k] = clamp(val, lo, hi)
            else: 
                x_new[k] = random.uniform(lo, hi) # 랜덤 생성
        m_new = evaluate_lt(model, scaler, x_new)
        f_new = objective(m_new,scaler) + penalty(m_new, rho)

        # 새롭게 만든 F,L,C 조합의 적응도가 HM의 가장 적응도가 높은 것보다 낮으면
        # 적응도가 낮아야 좋은것
        if f_new < HM_fit[-1]:
            # 변경
            HM[-1] = x_new
            HM_metrics[-1] = m_new
            HM_fit[-1] = f_new

            # 적응도가 오름차순으로 HM, HM_metrics, HM_fit 정렬
            idx = sorted(range(hms), key = lambda i : HM_fit[i])
            HM = [HM[i] for i in idx]
            HM_metrics = [HM_metrics[i] for i in idx]
            HM_fit = [HM_fit[i] for i in idx]
        
        if HM_fit[0] < best_f:
            best_x = HM[0]
            best_m = HM_metrics[0]
            best_f = HM_fit[0]

        
        history.append(best_f)
        print(f"[{it+1} / {max_iters}] best_f = {best_f:.4g}, best_m={best_m}")
        # .4g : 유효한 숫자 4자리
    return best_x, best_m, best_f, history


if __name__ == "__main__":
    bx, bm, bf, hist = harmony_search(max_iters= 1000)

    print("\n=== FINAL RESULT ===")
    print("Besy Designed:", bx)
    print("Mettrics:", bm)
    print("Fitness:", bf)
