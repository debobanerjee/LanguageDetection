import logging
from logging.handlers import RotatingFileHandler

def get_logger_handler(experiment: int):
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')
    my_handler = RotatingFileHandler(filename=f"output/logs/experiment-{str(experiment)}/language_detection.log", mode='a', backupCount=5, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.INFO)
    return my_handler

def get_logger_handler_data_preprocessing():
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')
    my_handler = RotatingFileHandler(filename=f"output/logs/data-preprocessing/preprocessing.log", mode='a', backupCount=5, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.INFO)
    return my_handler