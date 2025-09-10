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
# rapidfuzz
# requests
# gcsfs
# google-cloud-storage
 
#runtime: daily_schedule
 

import base64
import json
from sqlalchemy import create_engine, text
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Import your custom functions
#from insert_data_into_tables import (load_temperature_pressure_data,insert_daily_temperature,insert_forecast_temperature)
from insert_data_into_tables import(load_stations_data, load_wildfires_data,load_temperature_pressure_data,insert_daily_temperature,
insert_forecast_temperature, insert_forecast_pressure,load_hydro_droughts_data,load_metero_droughts_data,load_indicator_categories_data)
    
## entry point function
def daily_schedule(event, context):
    """
    Cloud Function triggered by Pub/Sub for daily jobs.
    """
    if 'data' in event:
        payload = json.loads(base64.b64decode(event['data']).decode('utf-8'))
        print(payload)
        
        logging.info("Daily schedule triggered")

        try:
            DB_USER = "postgres"
            DB_PASS = "Database%40123"
            DB_HOST = "34.100.141.55"
            DB_PORT = "5432"
            # DB_NAME = "HO_IFRC_ARG"
            DB_NAME = "ho_ifrc_arg"
        
            if not all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
                raise ValueError("One or more DB environment variables are missing")

            DB_URL = (
                f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
                f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )
            engine = create_engine(DB_URL)

            # Run your data pipeline
            
            ## run all first to force run -----------
            # load_stations_data()
            # load_wildfires_data()
            # load_temperature_pressure_data()
            # insert_daily_temperature()
            # insert_forecast_temperature()
            # insert_forecast_pressure() 
            load_hydro_droughts_data()  #--remaining
            load_metero_droughts_data() #--remaining
            # load_indicator_categories_data()
            
            # ## run daily schedule--------------
            # insert_daily_temperature()
            # insert_forecast_temperature()
            
            print("All Task Completed")

            logging.info("Daily job completed successfully")
            return "Success"

        except Exception as e:
            logging.error(f"Error during daily job: {e}")
            raise

