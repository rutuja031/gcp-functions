## cloud function for daily temp insert and forecast
### final working code

# requirements.txt:
# pandas
# requests
# rapidfuzz
# numpy
# sqlalchemy
# psycopg2-binary
# scikit-learn
# xgboost
 
#runtime: run_all_temp
 

import base64
import json
import requests
import zipfile
import io
import pandas as pd
from rapidfuzz import process
from datetime import datetime
import numpy as np
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Import your custom functions
from insert_data_into_tables import (load_temperature_pressure_data,insert_daily_temperature,insert_forecast_temperature)

# DATABASE CONNECTION
DB_USER = "postgres"
DB_PASS = "Database%40123"
DB_HOST = "34.100.141.55"
DB_PORT = "5432"
DB_NAME = "HO_IFRC_ARG"

DB_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DB_URL)     
    
def daily_schedule(event, context):
    """
    Cloud Function triggered by Pub/Sub for daily jobs.
    """
    logging.info("✅ Daily schedule triggered")

    try:
        load_temperature_pressure_data()
        insert_daily_temperature()
        insert_forecast_temperature()
        logging.info("✅ Daily job completed successfully")
        return "Success"
    except Exception as e:
        logging.error(f"❌ Error during daily job: {e}")
        raise
