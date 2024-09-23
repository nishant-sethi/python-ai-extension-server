import datetime
import logging
import os
from pathlib import Path

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s: line:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def time_taken(start_time):
    return f"{(datetime.datetime.now() - start_time).total_seconds():.3f}s "


def check_if_model_exists(model_name):
    if not model_name:
        logging.error(f"Model name not provided ")
        raise Exception(f"Model name not provided")
    home_dir = Path.home()
    registry_dir = os.getenv("REGISTRY_DIR")
    full_path = Path.joinpath(home_dir, registry_dir, 'library', model_name)
    return full_path.exists()


def log_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
