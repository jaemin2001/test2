import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./core/.env",verbose=True)

class Settings:
    DB_USERNAME : str = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST : str = os.getenv("DB_HOST","localhost")
    DB_PORT : str = os.getenv("DB_PORT",3306)
    DB_DATABASE : str = os.getenv("DB_DATABASE")
    DATABASE_URL = os.getenv("DATABASE_URL")
    # print(f"DATABSE_URL: {DATABASE_URL}")

     
