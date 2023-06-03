import os
from pathlib import Path
import environ
env = environ.Env()
BASE_DIR = f'{Path(__file__).resolve().parent.parent}\ML_Logical_Regression'
print(BASE_DIR)
environ.Env.read_env(os.path.join(BASE_DIR, '../../.env'))
DBNAME = env('DBNAME')
HOST = env('HOST')
USER = env('USER')
PASSWORD =  env('PASSWORD')
PORT = env('PORT')