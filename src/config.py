import os

from dotenv import load_dotenv

load_dotenv(".env")


class Config(object):
    RANDOM_STATE: int = int(os.environ.get("RANDOM_STATE"))
