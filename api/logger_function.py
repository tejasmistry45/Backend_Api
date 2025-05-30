import logging
import datetime
from datetime import datetime
import os
# from . import config
from dotenv import load_dotenv
load_dotenv()


def logger_function(name,e,n):
    # Handling Logs
    loggerfunc = logging.getLogger(name)
    loggerfunc.setLevel(logging.INFO)

    date = datetime.now().strftime('%m%d%Y')

    log_dir = os.getenv("LOGGER_FILE_PATH")
    if log_dir is None:
        raise ValueError("LOGGER_FILE_PATH environment variable is not set.")

    # Create directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{name}{date}.log")
    handlerfunc = logging.FileHandler(log_path, mode='a')
    formatterfunc = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    handlerfunc.setFormatter(formatterfunc)

    if not loggerfunc.hasHandlers():
        loggerfunc.addHandler(handlerfunc)

    if n == 1:
        loggerfunc.info(f"{e}")
    elif n == 2:
        loggerfunc.exception(f"{e}")
        loggerfunc.error(f"{e}")
        loggerfunc.critical(f"{e}")
