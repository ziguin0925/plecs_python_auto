# logger_config.py
import logging

def get_logger(name="sim_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # 중복 방지
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)

        # 파일 핸들러
        file_handler = logging.FileHandler('simulation.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger