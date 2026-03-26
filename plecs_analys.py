import xmlrpc.client as xml
import numpy as np 
import pandas as pd
import os 

opts_list = []
def SystemParameters(i):
    Param = {}
    Param['R1'] = 9+i
    Param['Kp'] = 0.1+i # PI_Digital1
    Param['mod_max'] = 10+i
    return Param

def LoadSimParameters(varin: dict):
    for k, _ in varin.items():
        varin[k] = float(varin[k])
    opts = {"ModelVars": varin}
    return opts



model = '260313 AC_DC_sin_PFC_JH4'
file_type = '.plecs'

plecs = xml.Server("http://localhost:1080/RPC2").plecs
plecs.load(r"C:\Users\KNU007\Documents\python_prac\plecs\260313 AC_DC_sin_PFC_JH4.plecs")
print(plecs.get(f'{model}/PI_Digital1'))
print(plecs.get(f'{model}/Scope6'))

opts1 = LoadSimParameters(SystemParameters(0))
opts2 = LoadSimParameters(SystemParameters(0))
opts_list.append(opts1)
opts_list.append(opts2)
result = plecs.simulate(model, opts_list)


# a = plecs.scope(f'{model}/Scope6', 'HoldTrace','Scope3_Data') # Scope6에 "Scope3_Data"라는 이름으로 그래프가 저장됨.


# 'Time', 'Values'

for i in range(len(result)):
    time = result[i]['Time']
    values = result[i]['Values']

    num_channels = len(values)

    data = pd.DataFrame()
    data['Time'] = time

    for j in range(num_channels):
        data[f'Value_{j}'] = values[j]

    data.to_csv(f'output_all_{j}.csv', index=False)

    max_capacitor = np.array(values[1]).max()

    print(values[1])
    print(max_capacitor)
