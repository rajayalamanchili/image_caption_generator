import logging

# set logging
log_format = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —"
                               "%(funcName)s:%(lineno)d — %(message)s")

# set log handler
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_format)

# remove previous handlers and add current handler
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers = []
    
logger.addHandler(log_handler)
logger.propagate = False

