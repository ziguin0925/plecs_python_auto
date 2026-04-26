import pandas as pd
import os
from logger_config import get_logger
import traceback
from multiprocessing import Lock, Manager

# 전역 Lock 생성
manager = Manager()
csv_lock = manager.Lock()

logger = get_logger()


OUT_BASE = "./out"
SAVE_FOLDER = "data_csv"
OUT_DIR = f"{OUT_BASE}/{SAVE_FOLDER}"

def rawdata_to_csv(result_data):
    '''
    plecs.simulate()의 return값을 바로 csv에 저장하고 싶을때
    '''
    for i in range(len(result_data)):
        time = result_data[i]['Time']
        values = result_data[i]['Values']

        num_channels = len(values)

        data = pd.DataFrame()
        data['Time'] = time

        for j in range(num_channels):
            data[f'Value_{j}'] = values[j]
            
        data.to_csv(f'output_all_{i}.csv', index=False)


def data_to_csv(results, out_dir = OUT_DIR):
    try:
        if not results:
            return

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "ripple_results.csv")

        with csv_lock:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    except Exception as e:
        logger.error(f"[CSV Insert Exception][Error: {e}]")
        traceback.print_exc()
        raise

