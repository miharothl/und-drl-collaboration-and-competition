import logging

DRL_LOG_NAME = "drl"

drl_logger = logging.getLogger(DRL_LOG_NAME)

################################################################################
# EPOCH

EPOCH = logging.INFO - 2
logging.addLevelName(EPOCH, 'EPOCH')
logging.EPOCH = EPOCH


def epoch(self, msg, extra={'params': None}):
    self.log(EPOCH, msg, extra=extra)


logging.Logger.epoch = epoch

################################################################################
# EPISODE

EPISODE = logging.INFO - 4
logging.addLevelName(EPISODE, 'EPISODE')
logging.EPISODE = EPISODE


def episode(self, msg, extra={'params': None}):
    self.log(EPISODE, msg, extra=extra)


logging.Logger.episode = episode

################################################################################
# STEP

STEP = logging.INFO - 6
logging.addLevelName(STEP, 'STEP')
logging.STEP = STEP


def step(self, msg, extra={'params': None}):
    self.log(STEP, msg, extra=extra)


logging.Logger.step = step
