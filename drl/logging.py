import logging
import logging.config
import sys

DRL_LOG_NAME = "drl"


def init_logging(level=logging.ERROR):

    drl_log = logging.getLogger(DRL_LOG_NAME)
    drl_log.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(params)s')
    handler.setFormatter(formatter)
    drl_log.addHandler(handler)
    drl_log.propagate = False


def set_logging_level(level):
    logger = logging.getLogger(DRL_LOG_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def transform_verbose_count_to_logging_level(count):

    if count == 1:
        return 20
    elif count > 1:
        return 20 - 2 * (count-1)

    return 0
