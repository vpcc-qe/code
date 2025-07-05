import logging
import datetime
import os

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_log_{current_time}.txt'

    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
