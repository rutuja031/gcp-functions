import pandas as pd
from sqlalchemy import create_engine, text 
import unicodedata
import numpy as np
# import pmdarima as pm   #--------------------------------
import re
import os
import io
from psycopg2.extras import execute_values
import requests
import zipfile
from rapidfuzz import process
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from google.cloud import storage
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Normalize column names for a DataFrame
def normalize_columns(df):
    try:
        df.columns = [
            unicodedata.normalize("NFKD", col)
            .encode("ascii", errors="ignore")
            .decode("utf-8")
            .strip()
            .lower()
            .replace(" ", "_")
            for col in df.columns
        ]
        return df
    except Exception as e:
        print(f"Error normalizing columns: {e}")
        return df

def clean_coldata(name):
    try:
        if pd.isna(name):
            return name
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
        name = re.sub(r'[^A-Za-z0-9\s]', '', name)
        return re.sub(r'\s+', ' ', name).strip()
    except Exception as e:
        print(f"Error cleaning column data: {e}")
        return name

def load_stations_data():   
    try:
        st_datafile = "gs://datafiles_bucket/STATIONS _ estaciones smn PAIS.xlsx"  ## GCS path
        st_threshold = "gs://datafiles_bucket/HEAT_COLD WAVES - OLAS_CALOR-FRIO.xlsx" ## GCS path
        try:
            df_st = pd.read_excel(st_datafile, sheet_name='estaciones smn')[["NRO INT", "ESTACION", "PROVINCIA", "LAT ", "LONG", "ALT (m)"]].dropna()
            df_st.columns = ["station_code", "station_name", "province_name", "latitude", "longitude", "altitude"]
        except Exception as e:
            print(f"Error reading stations Excel file: {e}")
            return

        try:
            df_th = pd.read_excel(st_threshold, sheet_name='estaciones')[["omm_id", "p90_tmax", "p90_tmin", "p10_tmax","p10_tmin"]].dropna()
            df_th.columns = ["station_code", "heat_max", "heat_min", "cold_max", "cold_min"]
        except Exception as e:
            print(f"Error reading threshold Excel file: {e}")
            return

        create_table_sql = """
        DROP TABLE IF EXISTS stg_stations;  
        CREATE TABLE IF NOT EXISTS stg_stations (
            station_code 	VARCHAR(20) PRIMARY KEY,
            station_name 	VARCHAR(255) NOT NULL,
            province_name 	VARCHAR(100) NOT NULL,
            latitude 		DOUBLE PRECISION,
            longitude 		DOUBLE PRECISION,
            altitude 		NUMERIC,
            cold_max 		NUMERIC,
            cold_min 		NUMERIC,
            heat_max 		NUMERIC,
            heat_min 		NUMERIC
        );
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))    
                conn.commit()
        except Exception as e:
            print(f"Error creating staging table: {e}")
            return

        try:
            df_st = df_st.merge(df_th, on="station_code", how="left")
            df_st = normalize_columns(df_st)
        except Exception as e:
            print(f"Error merging station dataframes: {e}")
            return

        try:
            df_st.to_sql("stg_stations", engine, if_exists="append", index=False)
        except Exception as e:
            print(f"Error writing to staging table: {e}")
            return

        insert_final_table_sql ="""
                INSERT INTO stations (station_code, station_name, province_code, 
                            latitude, longitude, altitude, cold_max, cold_min, heat_max, heat_min)
                SELECT station_code, station_name, (SELECT province_code FROM provinces WHERE UPPER(province_name) = UPPER(stg_stations.province_name)) AS province_code, 
                            latitude, longitude, altitude, cold_max, cold_min, heat_max, heat_min
                FROM stg_stations
            """
        try:
            with engine.begin() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM stg_stations"))
                count = result.scalar()
                print("Province Names:", df_st['province_name'].unique())
                if count > 0:
                    # Delete data from Stations before inserting the records
                    conn.execute(text("DELETE FROM stations"))
                    conn.execute(text(insert_final_table_sql ))
                conn.execute(text("TRUNCATE TABLE stg_stations"))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error inserting into final stations table: {e}")
            return

        print("Stations Data Inserted")
    except Exception as e:
        print(f"General error in load_stations_data: {e}")
        return
    
    # finally:
    #     print("Finalizing load_stations_data")

def load_wildfires_data():
    try:
        hectares_provincewise_url  = "https://ciam.ambiente.gob.ar/dt_csv.php?dt_id=409"
        try:
            df_hectares_provincewise = pd.read_csv(hectares_provincewise_url, sep=";", encoding="utf-8",low_memory=False)
            df_hectares_provincewise = df_hectares_provincewise[["jurisdicción","año_2017","año_2018","año_2019","año_2020","año_2021","año_2022","año_2023","año_2024"]].dropna()
            df_hectares_provincewise = normalize_columns(df_hectares_provincewise)
            df_hectares_provincewise = df_hectares_provincewise.iloc[1:]
            df_hectares_provincewise.columns = ["location", "year_2017", "year_2018", "year_2019", "year_2020", "year_2021", "year_2022", "year_2023", "year_2024"]
        except Exception as e:
            print(f"Error reading hectares CSV: {e}")
            return

        fires_provincewise_url  = "https://ciam.ambiente.gob.ar/dt_csv.php?dt_id=313"
        try:
            df_fires_provincewise = pd.read_csv(fires_provincewise_url, sep=";", encoding="utf-8",low_memory=False)[["jurisdicción","año_2017","año_2018","año_2019","año_2020","año_2021","año_2022","año_2023","año_2024","año_2025"]].dropna()
            df_fires_provincewise = normalize_columns(df_fires_provincewise)
            df_fires_provincewise = df_fires_provincewise.iloc[1:]
            df_fires_provincewise.columns = ["location", "year_2017", "year_2018", "year_2019", "year_2020", "year_2021", "year_2022", "year_2023", "year_2024", "year_2025"]
        except Exception as e:
            print(f"Error reading fires CSV: {e}")
            return

        try:
            df_fires = df_fires_provincewise.merge(df_hectares_provincewise, on=["location"], how="left")
            df_fires = normalize_columns(df_fires)
            print("Merged Fires Data:", df_fires)
        except Exception as e:
            print(f"Error merging fires dataframes: {e}")
            return

        # Remove the suffixes _x and _y from the column names
        fire_cols = [col for col in df_fires.columns if col.endswith("_x")]
        fires_long = df_fires.melt(id_vars=["location"], value_vars=fire_cols, var_name="year", value_name="fire_count")
        hectares_cols = [col for col in df_fires.columns if col.endswith("_y")]
        hectares_long = df_fires.melt(id_vars=["location"], value_vars=hectares_cols, var_name="year", value_name="hectares")
        # Remove the year from the column names and add only the number value of the year
        fires_long["year"] = fires_long["year"].str.extract(r'(\d{4})')
        hectares_long["year"] = hectares_long["year"].str.extract(r'(\d{4})')
        
        df_fires_all = fires_long.merge(hectares_long,on=["location", "year"], how="left")
        df_fires_all["year"] = df_fires_all["year"].astype(int)
        print(df_fires_all)
        df_fires_all['fire_count'] = pd.to_numeric(df_fires_all['fire_count'].replace("s/d", np.nan), errors='coerce')
        df_fires_all['location'] = df_fires_all['location'].apply(clean_coldata)
        df_fires_all = df_fires_all[~df_fires_all['location'].str.contains("CABA", case=False)]

        create_table_sql = """
            DROP TABLE IF EXISTS stg_fires_by_location;  
            CREATE TABLE IF NOT EXISTS stg_fires_by_location (
            location           	VARCHAR(20) NOT NULL,
            year				INTEGER NOT NULL,
            hectares			NUMERIC,
            fire_count			INTEGER
            );
            """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))    
                conn.commit()
        except Exception as e:
            print(f"Error creating fires staging table: {e}")
            return

        try:
            df_fires_all.to_sql("stg_fires_by_location", engine, if_exists="append", index=False)
        except Exception as e:
            print(f"Error writing fires data to staging table: {e}")
            return

        insert_final_table_sql = """
            INSERT INTO fires_by_location (region_code, province_code, year, hectares, fire_count, measurement_type)
            SELECT (select region_code from provinces 
                    where UPPER(province_name) = UPPER(stg_fires_by_location.location)),
                    (select province_code from provinces 
                    where UPPER(province_name) = UPPER(stg_fires_by_location.location)),
                    year, hectares, fire_count, 'A'
            FROM stg_fires_by_location
        """
        try:
            with engine.begin() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM stg_fires_by_location"))
                count = result.scalar()
                if count > 0:
                    conn.execute(text("DELETE FROM fires_by_location"))
                    conn.execute(text(insert_final_table_sql))
                conn.execute(text("TRUNCATE TABLE stg_fires_by_location"))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error inserting into final fires_by_location table: {e}")
            return
        
         # Insert data into fires_by_months
        
        fires_monthwise_url  = "https://ciam.ambiente.gob.ar/dt_csv.php?dt_id=312"
        try:
            df_fires_monthwise = pd.read_csv(fires_monthwise_url, sep=";", encoding="utf-8",low_memory=False)            
            df_fires_monthwise = df_fires_monthwise[["mes","año_2017","año_2018","año_2019","año_2020","año_2021","año_2022","año_2023","año_2024","año_2025"]]  #.dropna()
            df_fires_monthwise = normalize_columns(df_fires_monthwise)
            df_fires_monthwise = df_fires_monthwise.iloc[1:]
            df_fires_monthwise.columns = [ "month", "year_2017", "year_2018", "year_2019", "year_2020", "year_2021", "year_2022", "year_2023", "year_2024", "year_2025"]
        except Exception as e:
            print(f"Error reading fires monthwise CSV: {e}")
            return
        # all twelve months for the original dataframe should be included in the melted dataframe
        df_fires_monthwise = df_fires_monthwise.melt(id_vars=["month"], value_vars=["year_2017", "year_2018", "year_2019", "year_2020", "year_2021", "year_2022", "year_2023", "year_2024", "year_2025"], var_name="year", value_name="fire_count")
        df_fires_monthwise["year"] = df_fires_monthwise["year"].str.extract(r'(\d{4})')
        print("Fires Monthwise Data2:", df_fires_monthwise)
        # convert month names to month numbers
        df_fires_monthwise["month"] = df_fires_monthwise["month"].str.lower().str.strip()
        month_mapping = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12
        }
        df_fires_monthwise["month"] = df_fires_monthwise["month"].map(month_mapping)
        # If the month is not in the mapping, set it to NaN
        df_fires_monthwise["month"] = df_fires_monthwise["month"].where(df_fires_monthwise["month"].notna(), np.nan)
        
        create_table_sql = """
            DROP TABLE IF EXISTS stg_fires_by_months;  
            CREATE TABLE IF NOT EXISTS stg_fires_by_months (
            year				INTEGER NOT NULL,
	        month				INTEGER NOT NULL,
	        fire_count			INTEGER
            );
            """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))    
                conn.commit()
        except Exception as e:
            print(f"Error creating fires monthwise staging table: {e}")
            return

        try:
            df_fires_monthwise.to_sql("stg_fires_by_months", engine, if_exists="append", index=False)
        except Exception as e:
            print(f"Error writing fires monthwise data to staging table: {e}")
            return
        
        # Insert data into final fires_by_months
        insert_final_table_sql = """
            INSERT INTO fires_by_months (year, month_no, fire_count, measurement_type)
            SELECT year, month, fire_count, 'A'
            FROM stg_fires_by_months
        """
        try:
            with engine.begin() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM stg_fires_by_months"))
                count = result.scalar()
                if count > 0:
                    conn.execute(text("DELETE FROM fires_by_months"))
                    conn.execute(text(insert_final_table_sql))
                conn.execute(text("TRUNCATE TABLE stg_fires_by_months"))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error inserting into final fires_by_months table: {e}")
            return
    # finally:
    #     print("Finalizing load_fires_data")
    except Exception as e:
            print(f"General error in load_wildfires_data: {e}")
            return

# def insert_fires_by_months_forecast():
#     try:
#         try:
#             query_months = "SELECT DISTINCT month_no FROM fires_by_months ORDER BY month_no"
#             df_months = pd.read_sql(query_months, engine)
#             months = df_months['month_no'].tolist()
#         except Exception as e:
#             print(f"Error loading months: {e}")
#             return

#         for month in months:
#             try:
#                 query_data = """
#                     SELECT year, fire_count
#                     FROM fires_by_months
#                     WHERE month_no = %s AND measurement_type = 'A'
#                     ORDER BY year
#                 """
#                 df = pd.read_sql(query_data, engine, params=(month,))
#             except Exception as e:
#                 print(f"Error loading fire data for month {month}: {e}")
#                 continue

#             if len(df) < 3:
#                 print(f"Skipping month {month}: Not enough data")
#                 continue

#             values = df['fire_count'].dropna().values
#             if len(values) < 3:
#                 print(f"Skipping month {month}: Not enough non-NaN data")
#                 continue

#             if np.all(values == values[0]):
#                 print(f"Skipping month {month}: constant values")
#                 continue

#             last_year = int(df['year'].max())

#             try:
#                 model = pm.auto_arima(values, seasonal=False, suppress_warnings=True, error_action='ignore')
#                 forecast = model.predict(n_periods=1)[0]
#                 forecast = round(forecast, 1)  # Optional: round to 1 decimal
#             except Exception as e:
#                 print(f"Skipping month {month}: model fitting error {e}")
#                 continue

#             #print(f"Month {month} Year {last_year + 1} Forecast: {forecast}")

#             try:
#                 insert_query = text("""
#                     INSERT INTO fires_by_months (year, month_no, fire_count, measurement_type)
#                     VALUES (:year, :month_no, :fire_count, 'F')
#                 """)
#                 with engine.begin() as conn:
#                     conn.execute(insert_query, {
#                         "year": int(last_year + 1),
#                         "month_no": int(month),
#                         "fire_count": float(forecast)
#                     })
#             except Exception as e:
#                 print(f"Error inserting forecast for month {month}: {e}")
#                 continue

#     except Exception as e:
#         print(f"Fatal error in insert_fires_by_months_forecast: {e}")

# def insert_fires_by_location_forecast():
#     try:
#         try:
#             query_provinces = "SELECT province_code FROM provinces ORDER BY province_name"
#             df_provinces = pd.read_sql(query_provinces, engine)
#             province_codes = df_provinces['province_code'].tolist()
#         except Exception as e:
#             print(f"Error loading provinces: {e}")
#             return

#         for province_code in province_codes:
#             try:
#                 query_data = """
#                     SELECT year, fire_count, hectares
#                     FROM fires_by_location
#                     WHERE province_code = %s AND measurement_type = 'A'
#                     ORDER BY year
#                 """
#                 df = pd.read_sql(query_data, engine, params=(province_code,))
#             except Exception as e:
#                 print(f"Error loading fire data for province {province_code}: {e}")
#                 continue

#             if len(df) < 3:
#                 print(f"Skipping province {province_code}: not enough data")
#                 continue

#             years = df['year'].values
#             fire_counts = df['fire_count'].dropna().values
#             hectares_vals = df['hectares'].dropna().values

#             # Forecast fire_count
#             fire_forecast = None
#             if len(fire_counts) >= 3 and not np.all(fire_counts == fire_counts[0]):
#                 try:
#                     model_fire = pm.auto_arima(fire_counts, seasonal=False, suppress_warnings=True, error_action='ignore')
#                     fire_forecast = round(model_fire.predict(n_periods=1)[0], 1)
#                 except Exception as e:
#                     print(f"Skipping fire_count forecast for province {province_code}: {e}")

#             # Forecast hectares
#             hectares_forecast = None
#             if len(hectares_vals) >= 3 and not np.all(hectares_vals == hectares_vals[0]):
#                 try:
#                     model_hectares = pm.auto_arima(hectares_vals, seasonal=False, suppress_warnings=True, error_action='ignore')
#                     hectares_forecast = round(model_hectares.predict(n_periods=1)[0], 1)
#                 except Exception as e:
#                     print(f"Skipping hectares forecast for province {province_code}: {e}")

#             forecast_year = int(years[-1]) + 1

#             if fire_forecast is not None and hectares_forecast is not None:
#                 try:
#                     query_region = "SELECT region_code FROM provinces WHERE province_code = %s"
#                     df_region = pd.read_sql(query_region, engine, params=(province_code,))
#                     if df_region.empty:
#                         print(f"Region code not found for province {province_code}")
#                         continue
#                     region_code = df_region.iloc[0]['region_code']

#                     insert_query = text("""
#                         INSERT INTO fires_by_location (region_code, province_code, year, fire_count, hectares, measurement_type, created_by, created_at)
#                         VALUES (:region_code, :province_code, :year, :fire_count, :hectares, 'F', 'system', :created_at)
#                     """)
#                     with engine.begin() as conn:
#                         conn.execute(insert_query, {
#                             "region_code": region_code,
#                             "province_code": province_code,
#                             "year": forecast_year,
#                             "fire_count": fire_forecast,
#                             "hectares": hectares_forecast,
#                             "created_at": datetime.now()
#                         })
#                     print(f"Inserted forecast for province {province_code}, year {forecast_year}")
#                 except Exception as e:
#                     print(f"Error inserting forecast for province {province_code}: {e}")
#             else:
#                 print(f"Incomplete forecast for province {province_code}, nothing inserted.")

#     except Exception as e:
#         print(f"Fatal error in insert_fires_by_location_forecast: {e}")

def load_temperature_pressure_data():
    try:
        mintemp_datafile= "gs://datafiles_bucket/TEMP-HUMIDITY-PRESION-WIND - Exp. 208486-2.txt"
        try:
            df_temperature_f2 = pd.read_csv(mintemp_datafile)[["Estacion","Fecha","Temp. Maxima (°C)","Temp. Minima (°C)","Presion media Nivel Estacion (hPa)","Precipitacion (mm)","Hum. Relativa Media (%)","Dirección viento maximo","Velocidad viento maximo (m/s)"]].dropna()
            df_temperature_f2.columns = ["station_code", "daily_date", "max_temp", "min_temp", "pressure", "precipitation", "humidity", "max_wind_direction", "max_wind_speed"]
            df_temperature_f2 = normalize_columns(df_temperature_f2)

            df_temp_pres = df_temperature_f2 #pd.concat([df_temperature_f1, df_temperature_f2])
            #replace \N with None
            df_temp_pres = df_temp_pres.replace(r'\\N', None, regex=True)
            print("Temperature Data:", df_temp_pres.head())
            print("Pressure Data:", df_temp_pres.tail())

        except Exception as e:
            print(f"Error reading temperature data files: {e}")
            return
        
        create_table_sql = """
        DROP TABLE IF EXISTS stg_temperature_pressure;  
        CREATE TABLE IF NOT EXISTS stg_temperature_pressure (        
            station_code 		VARCHAR(20) NOT NULL,
            daily_date 			DATE NOT NULL,
            min_temp 			NUMERIC,
            max_temp 			NUMERIC,
            pressure 			NUMERIC,
            precipitation		NUMERIC,
            humidity			NUMERIC,
            max_wind_direction	NUMERIC,
            max_wind_speed		NUMERIC,
            PRIMARY KEY (station_code, daily_date)
        );
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))    
                conn.commit()
        except Exception as e:
            print(f"Error creating staging table: {e}")
            return
        
        try:
            df_temp_pres.to_sql("stg_temperature_pressure", engine, if_exists="append", index=False)
            print("Temperature and Pressure data loaded successfully.")
        except Exception as e:
            print(f"Error writing to staging table: {e}")
            return

        insert_final_temp_table_sql ="""
                INSERT INTO temperature (daily_date, station_code, min_temp, max_temp, 
                            measurement_type, risk_level)
                SELECT daily_date, station_code, min_temp, max_temp, 'A' AS measurement_type, 
                        (SELECT CASE 
                        WHEN min_temp < (cold_min - 5) THEN 'L'
                        WHEN max_temp > (heat_max + 5) THEN 'H'
                        ELSE 'M'
                        END:: risk_level_enum FROM stations where station_code = stg_temperature_pressure.station_code) AS risk_level
                        FROM stg_temperature_pressure
                        WHERE daily_date >= (
                            (SELECT MAX(daily_date) FROM stg_temperature_pressure) - INTERVAL '2 years'
                        )
            """
        
        insert_final_pressure_table_sql ="""
                INSERT INTO pressure (daily_date, station_code, pressure, measurement_type)
                SELECT daily_date, station_code, pressure, 'A' AS measurement_type FROM stg_temperature_pressure
                WHERE daily_date >= (
                    (SELECT MAX(daily_date) FROM stg_temperature_pressure) - INTERVAL '2 years'
                )
            """
         # Insert data into climate_months table
        insert_climate_months_table_sql ="""
                INSERT INTO climate_months (station_code, month_num, month_name, avg_min_temp, avg_max_temp, avg_pressure)
                SELECT station_code, EXTRACT(MONTH FROM daily_date) AS month_num,
                    TRIM(TO_CHAR(daily_date, 'Month')) AS month_name,
                    ROUND(AVG(min_temp), 2) AS avg_min_temp, ROUND(AVG(max_temp), 2) AS avg_max_temp,
                    ROUND(AVG(pressure), 2) AS avg_pressure 
                FROM stg_temperature_pressure
                WHERE EXTRACT(YEAR FROM daily_date) >= (
                (SELECT EXTRACT(YEAR FROM MAX(daily_date)) FROM stg_temperature_pressure) -2)
                GROUP BY month_num, month_name, station_code
                ORDER BY station_code, month_num ASC;
                """

        try:
            with engine.begin() as conn:
                # Check if there are any records in the staging table
                check_record_sql = """
                SELECT COUNT(*) FROM stg_temperature_pressure 
                WHERE daily_date >= ((SELECT MAX(daily_date) 
                                    FROM stg_temperature_pressure) - INTERVAL '2 years')
                """
                result = conn.execute(text(check_record_sql))
                count = result.scalar()
                print("Records to be inserted from staging table stg_temperature_pressure:", count)
                if count > 0:
                    # Delete data from temperature before inserting the records
                    conn.execute(text("DELETE FROM temperature"))
                    conn.execute(text(insert_final_temp_table_sql))
                    conn.commit()
        
        # Delete data from pressure before inserting the records
            with engine.begin() as conn:
                if count > 0:
                    conn.execute(text("DELETE FROM pressure"))
                    conn.execute(text(insert_final_pressure_table_sql))
                    conn.commit()

        # Insert data into pressure by months table
            with engine.begin() as conn:
                if count > 0:
                    conn.execute(text("DELETE FROM climate_months"))
                    conn.execute(text(insert_climate_months_table_sql))
                    conn.execute(text("TRUNCATE TABLE stg_temperature_pressure"))
                    conn.commit()

        except Exception as e:
            print(f"Error inserting into final stations table: {e}")
            return

        print("Temperature, Climate months and Pressure Data Inserted")

    except Exception as e:
        print(f"General error in load_temperature_pressure_data: {e}")
        return

    # finally:
    #     print("Finalizing load_temperature_pressure_data")
        
def insert_daily_temperature():  ####### for daily temperature
    try:
        print("Starting daily temperature data fetch...")
        url = "https://ssl.smn.gob.ar/dpd/zipopendata.php?dato=regtemp"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            txt_filename = [name for name in z.namelist() if name.endswith('.txt')][0]
            with z.open(txt_filename) as txt_file:
                df = pd.read_fwf(txt_file, encoding='latin1')

        df.columns = df.columns.str.strip()
        df = df[df['FECHA'] != '--------'].copy()
        df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d%m%Y', errors='coerce')
        df['NOMBRE'] = df['NOMBRE'].str.strip()
        df['TMAX'] = pd.to_numeric(df['TMAX'], errors='coerce')
        df['TMIN'] = pd.to_numeric(df['TMIN'], errors='coerce')

        df = df.rename(columns={
            'FECHA': 'daily_date',
            'NOMBRE': 'station_name',
            'TMAX': 'max_temp',
            'TMIN': 'min_temp'
        })
        df = df[['station_name', 'daily_date', 'min_temp', 'max_temp']]
        df = df.dropna(subset=['daily_date', 'station_name'])

        # Fetch station mapping
        with engine.connect() as conn:
            station_map = dict(conn.execute(text("SELECT station_code, station_name FROM stations")).fetchall())

        name_to_code = {v.strip(): k for k, v in station_map.items()}
        known_names = list(name_to_code.keys())

        manual_overrides = {
            "LA QUIACA OBSERVATORIO": "LA QUIACA OBS.",
            "OBERA": "OBERA AERO",
            "VILLA DE MARIA DEL RIO SECO": "VILLA MARIA DEL RIO SECO",
            "ESCUELA DE AVIACION MILITAR AERO": "ESC.AVIACION MILITAR AERO",
            "PILAR OBSERVATORIO": "PILAR OBS.",
            "VENADO TUERTO AERO": "VENADO TUERTO",
            "SAN FERNANDO AERO": "SAN FERNANDO",
            "LAS FLORES": "LAS FLORES AERO",
            "BUENOS AIRES OBSERVATORIO": "BUENOS AIRES",
            "BENITO JUAREZ": "BENITO JUAREZ AERO"
        }

        def resolve_station_code(name):
            if name in name_to_code:
                return name_to_code[name]
            if name in manual_overrides:
                canonical = manual_overrides[name]
                return name_to_code.get(canonical)
            result = process.extractOne(name, known_names, score_cutoff=90)
            if result:
                return name_to_code.get(result[0])
            return None

        df['station_code'] = df['station_name'].apply(resolve_station_code)
        df = df.dropna(subset=['station_code'])

        # Fetch thresholds
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT station_code, cold_min, heat_max FROM stations")).fetchall()
        thresholds = {code: {'cold_min': cmin, 'heat_max': hmax} for code, cmin, hmax in rows}

        def compute_risk(min_temp, max_temp, station_code):
            if pd.isna(min_temp) or pd.isna(max_temp) or station_code not in thresholds:
                return "M"
            cold_min = thresholds[station_code]['cold_min']
            heat_max = thresholds[station_code]['heat_max']
            if cold_min is None or heat_max is None:
                return "M"
            if min_temp < (cold_min - 5):
                return "L"
            elif max_temp > (heat_max + 5):
                return "H"
            return "M"

        df['risk_level'] = df.apply(
            lambda row: compute_risk(row['min_temp'], row['max_temp'], row['station_code']),
            axis=1
        )
        df['measurement_type'] = 'A'
        df['created_at'] = datetime.now()

        records = df[['daily_date', 'station_code', 'min_temp', 'max_temp',
                      'risk_level', 'measurement_type', 'created_at']].to_dict(orient="records")

        upsert_sql = """
        INSERT INTO temperature (daily_date, station_code, min_temp, max_temp,
                                 risk_level, measurement_type, created_at)
        VALUES (:daily_date, :station_code, :min_temp, :max_temp,
                :risk_level, :measurement_type, :created_at)
        ON CONFLICT (daily_date, station_code)
        DO UPDATE SET
            min_temp = EXCLUDED.min_temp,
            max_temp = EXCLUDED.max_temp,
            risk_level = EXCLUDED.risk_level,
            measurement_type = EXCLUDED.measurement_type,
            created_at = EXCLUDED.created_at
        """

        with engine.begin() as conn:
            conn.execute(text(upsert_sql), records)

        print(f"Inserted/Updated {len(records)} daily records into temperature table.")

    except Exception as e:
        print(f"Error in insert_daily_temperature: {e}")
        
def insert_forecast_temperature():   ## for daily
    try:
        forecast_days = 7
        lags = 7

        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }

        station_codes = pd.read_sql("SELECT DISTINCT station_code FROM temperature", engine)['station_code'].dropna().tolist()
        all_forecasts = []

        for station_code in station_codes:
            station_forecast = {'station_code': station_code, 'forecast': {}}

            for target_col in ['max_temp', 'min_temp']:
                try:
                    query = f"""
                        SELECT daily_date, station_code, {target_col}
                        FROM (
                            SELECT daily_date, station_code, {target_col}
                            FROM temperature
                            WHERE station_code = %s AND measurement_type = 'A'
                            ORDER BY daily_date DESC
                        ) sub
                        ORDER BY daily_date ASC
                    """
                    df = pd.read_sql(query, engine, params=(station_code,))
                    if df.empty:
                        continue

                    df['daily_date'] = pd.to_datetime(df['daily_date'])
                    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                    if df[target_col].isna().sum() > 20:
                        continue
                    df[target_col] = df[target_col].ffill()
                    df.set_index('daily_date', inplace=True)

                    for lag in range(1, lags + 1):
                        df[f'lag_{lag}'] = df[target_col].shift(lag)
                    df.dropna(inplace=True)
                    if len(df) < lags:
                        continue

                    lag_cols = [f'lag_{i}' for i in range(1, lags + 1)]
                    X, y = df[lag_cols], df[target_col]

                    split_index = int(len(df) * 0.8)
                    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                    best_model, best_mape = None, float('inf')
                    for model_name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        nonzero_actual = y_test != 0
                        if nonzero_actual.sum() == 0:
                            continue
                        mape = np.mean(np.abs((y_test[nonzero_actual] - y_pred[nonzero_actual]) / y_test[nonzero_actual])) * 100
                        if mape < best_mape:
                            best_mape = mape
                            best_model = model

                    if not best_model:
                        continue

                    last_known = df.iloc[-1:]
                    lags_list = [last_known[target_col].values[0]] + last_known[lag_cols[:-1]].values.flatten().tolist()

                    preds, dates = [], []
                    for i in range(forecast_days):
                        input_arr = np.array(lags_list[-lags:]).reshape(1, -1)
                        pred = best_model.predict(input_arr)[0]
                        next_date = df.index[-1] + pd.Timedelta(days=i + 1)
                        preds.append(pred)
                        dates.append(next_date)
                        lags_list.append(pred)

                    station_forecast['forecast'][target_col] = pd.Series(preds, index=dates)

                except Exception as e:
                    print(f"Error processing {target_col} for station {station_code}: {e}")
                    continue

            if 'min_temp' in station_forecast['forecast'] and 'max_temp' in station_forecast['forecast']:
                merged = pd.DataFrame({
                    'daily_date': station_forecast['forecast']['min_temp'].index,
                    'station_code': station_code,
                    'min_temp': station_forecast['forecast']['min_temp'].values,
                    'max_temp': station_forecast['forecast']['max_temp'].values
                })

                thresholds = pd.read_sql(
                    "SELECT cold_min, heat_max FROM stations WHERE station_code = %s",
                    engine,
                    params=(station_code,)
                )
                cold_min = thresholds.iloc[0]['cold_min'] if not thresholds.empty else None
                heat_max = thresholds.iloc[0]['heat_max'] if not thresholds.empty else None

                def classify_risk(row):
                    if pd.notnull(heat_max) and row['max_temp'] > heat_max + 5:
                        return 'H'
                    elif pd.notnull(cold_min) and row['min_temp'] < cold_min - 5:
                        return 'L'
                    return 'M'

                merged['risk_level'] = merged.apply(classify_risk, axis=1)
                merged['measurement_type'] = 'F'
                merged['min_temp'] = merged['min_temp'].round(1)
                merged['max_temp'] = merged['max_temp'].round(1)
                merged['created_at'] = datetime.now()

                all_forecasts.append(merged)

        if all_forecasts:
            full_df = pd.concat(all_forecasts, ignore_index=True)
            records = full_df.to_dict(orient="records")

            insert_query = """
            INSERT INTO temperature (daily_date, station_code, min_temp, max_temp,
                                     risk_level, measurement_type, created_at)
            VALUES (:daily_date, :station_code, :min_temp, :max_temp,
                    :risk_level, :measurement_type, :created_at)
            ON CONFLICT (daily_date, station_code) DO UPDATE
            SET min_temp = EXCLUDED.min_temp,
                max_temp = EXCLUDED.max_temp,
                risk_level = EXCLUDED.risk_level,
                measurement_type = EXCLUDED.measurement_type,
                created_at = EXCLUDED.created_at;
            """

            with engine.begin() as conn:
                conn.execute(text(insert_query), records)

            print("Forecasts inserted into temperature table.")
        else:
            print("No forecasts generated — check data availability.")

    except Exception as e:
        print(f"Error in insert_forecast_temperature: {e}")

def insert_forecast_pressure():  ## for daily
    try:
        forecast_days = 7
        lags = 7

        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }
        station_codes = pd.read_sql("SELECT DISTINCT station_code FROM pressure", engine)['station_code'].dropna().tolist()
        all_forecasts = []

        for station_code in station_codes:
            try:
                query = f"""
                SELECT daily_date, station_code, pressure
                FROM (
                    SELECT daily_date, station_code, pressure
                    FROM pressure
                    WHERE station_code = %s AND measurement_type = 'A'
                    ORDER BY daily_date DESC
                ) sub
                ORDER BY daily_date ASC
                """
                df = pd.read_sql(query, engine, params=(station_code,))
                
                if df.empty:
                    print(f"Skipping {station_code} — no data for pressure")
                    continue

                df['daily_date'] = pd.to_datetime(df['daily_date'])
                df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
                missing_count = df['pressure'].isna().sum()

                if missing_count > 20:
                    print(f"Skipping {station_code} — too many missing values in pressure ({missing_count})")
                    continue
                elif missing_count > 0:
                    df['pressure'] = df['pressure'].ffill()

                df.set_index('daily_date', inplace=True)

                # --- Create lag features ---
                for lag in range(1, lags + 1):
                    df[f'lag_{lag}'] = df['pressure'].shift(lag)
                df.dropna(inplace=True)

                if len(df) < lags:
                    continue

                lag_cols = [f'lag_{i}' for i in range(1, lags + 1)]
                X = df[lag_cols]
                y = df['pressure']
                split_index = int(len(df) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                best_model = None
                best_mape = float('inf')

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    nonzero_actual = y_test != 0
                    if nonzero_actual.sum() == 0:
                        continue
                    mape = np.mean(np.abs((y_test[nonzero_actual] - y_pred[nonzero_actual]) / y_test[nonzero_actual])) * 100

                    if mape < best_mape:
                        best_mape = mape
                        best_model = model

                if not best_model:
                    continue

                last_known = df.iloc[-1:]
                lags_list = [last_known['pressure'].values[0]] + last_known[lag_cols[:-1]].values.flatten().tolist()

                preds = []
                dates = []

                for i in range(forecast_days):
                    input_arr = np.array(lags_list[-lags:]).reshape(1, -1)
                    pred = best_model.predict(input_arr)[0]
                    pred = round(pred, 1)
                    next_date = df.index[-1] + pd.Timedelta(days=i + 1)
                    preds.append(pred)
                    dates.append(next_date)
                    lags_list.append(pred)

                forecast_df = pd.DataFrame({
                    'daily_date': dates,
                    'station_code': station_code,
                    'pressure': preds,
                    'measurement_type': 'F'
                })

                all_forecasts.append(forecast_df)

            except Exception as e:
                print(f"Error processing pressure for station {station_code}: {e}")
                continue

        if all_forecasts:
            full_df = pd.concat(all_forecasts, ignore_index=True)

            insert_query = """
            INSERT INTO pressure (daily_date, station_code, pressure, measurement_type, created_at)
            VALUES (:daily_date, :station_code, :pressure, :measurement_type, :created_at)
            ON CONFLICT (daily_date, station_code) DO UPDATE
            SET pressure = EXCLUDED.pressure,
                measurement_type = EXCLUDED.measurement_type,
                created_at = EXCLUDED.created_at;
            """

            with engine.begin() as conn:
                for _, row in full_df.iterrows():
                    conn.execute(text(insert_query), {
                        'daily_date': row['daily_date'],
                        'station_code': row['station_code'],
                        'pressure': row['pressure'],
                        'measurement_type': row['measurement_type'],
                        'created_at': datetime.now()
                    })

            print("Pressure forecasts inserted into pressure table with measurement_type = 'F'.")
        else:
            print("No pressure forecasts generated—check for missing or insufficient data.")

    except Exception as e:
        print(f"Error in insert_forecast_pressure: {e}")

def load_hydro_data_to_database(file_path):
    try:
        # Read CSV and select required columns
        df_hydro = pd.read_csv(file_path)[["cod_est", "fecha", "valor_indice"]].dropna()
        df_hydro.columns = ["station_code", "daily_date", "drought_index"]

        # Normalize column data (user-defined function)
        df_hydro = normalize_columns(df_hydro)

        # Print sample data for debugging
        #print("Hydro Drought Data Sample:\n", df_hydro.head())

        # SQL to create staging table
        create_table_sql = """
            DROP TABLE IF EXISTS stg_hydro_droughts;  
            CREATE TABLE IF NOT EXISTS stg_hydro_droughts (        
                station_code 	VARCHAR(20) NOT NULL,
                daily_date 		DATE NOT NULL,
                drought_index	NUMERIC,
                PRIMARY KEY (station_code, daily_date)
            );
        """

        # Create staging table and insert data
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
                df_hydro.to_sql("stg_hydro_droughts", conn, if_exists="append", index=False)
        except Exception as e:
            print(f"Error loading hydro drought data to staging table: {e}")
            return

        # SQL to insert data into final table
        insert_final_hydro_table_sql = """
            INSERT INTO hydrological_droughts (daily_date, station_code, value_index, measurement_type, risk_level)
            SELECT s.daily_date, s.station_code, s.drought_index, 'A' AS measurement_type,
                   t.category::drought_risk_level_enum AS risk_level
            FROM stg_hydro_droughts s
            JOIN spi_thresholds t 
              ON s.drought_index BETWEEN t.min_value AND t.max_value;
        """

        # Insert into final table only if staging has data
        try:
            with engine.begin() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM stg_hydro_droughts"))
                count = result.scalar()
                #print(f"Staging table row count: {count}")

                if count > 0:
                    conn.execute(text(insert_final_hydro_table_sql))
                    conn.commit()
                    print("Hydro drought data successfully inserted into final table.")
                else:
                    print("No data in staging table. Skipping final insert.")

        except Exception as e:
            print(f"Error inserting into hydrological_droughts table: {e}")

    except Exception as e:
        print(f"General error in load_hydro_data_to_database: {e}")
    # finally:
    #     print("Finalizing load_hydro_data_to_database")

def load_hydro_droughts_data():
    try:
        # Delete existing data from final table before new load
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM hydrological_droughts"))
            #print("Deleted existing records from hydrological_droughts")

        # GCS bucket and directory prefix
        bucket_name = "datafiles_bucket"
        prefix = "downloads_hydro/"

        # List all files in the GCS folder
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        print(f"Found {len(blobs)} files in GCS folder '{prefix}'")

        # Process each CSV file
        for blob in blobs:
            if blob.name.endswith(".csv"):
                #print("Processing file:", blob.name)

                # Download file to /tmp/ (Cloud Functions temp directory)
                temp_path = f"/tmp/{os.path.basename(blob.name)}"
                blob.download_to_filename(temp_path)
                #print(f"Downloaded {blob.name} to {temp_path}")

                # Load into database
                load_hydro_data_to_database(temp_path)

                # Remove temp file to free space
                os.remove(temp_path)
                #print(f"Removed temp file {temp_path}")

    except Exception as e:
        print(f"Error in load_hydro_droughts_data: {e}")

def load_metero_data_to_database(file_path):
    try:
        # Read the CSV and select required columns
        df_metero = pd.read_csv(file_path)[["omm_id", "fecha", "valor_indice", "pentada_fin"]].dropna()
        df_metero.columns = ["station_code", "monthly_date", "drought_index", "pentada_end"]

        # Optional: normalize data
        df_metero = normalize_columns(df_metero)

        # Print first few rows for debugging
        #print("Metero Drought Data Sample:\n", df_metero.head())

        # SQL to create staging table
        create_table_sql = """
            DROP TABLE IF EXISTS stg_metero_droughts;
            CREATE TABLE IF NOT EXISTS stg_metero_droughts (        
                station_code 	VARCHAR(20) NOT NULL,
                monthly_date 	DATE NOT NULL,
                drought_index	NUMERIC,
                pentada_end     INTEGER,
                PRIMARY KEY (station_code, monthly_date)
            );
        """

        # Create staging table and insert data
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
                df_metero.to_sql("stg_metero_droughts", conn, if_exists="append", index=False)
        except Exception as e:
            print(f"Error loading metero drought data to staging table: {e}")
            return

        # SQL to insert final data
        insert_final_metero_table_sql = """
            INSERT INTO meterological_droughts (monthly_date, station_code, value_index, measurement_type, risk_level)
            SELECT s.monthly_date, s.station_code, s.drought_index, 'A' AS measurement_type,
                   t.category::drought_risk_level_enum AS risk_level
            FROM stg_metero_droughts s 
            JOIN spi_thresholds t 
              ON s.drought_index >= t.min_value AND s.drought_index < t.max_value;
        """

        # Insert into final table if staging table has data
        try:
            with engine.begin() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM stg_metero_droughts"))
                count = result.scalar()
                #print(f"Staging table row count: {count}")

                if count > 0:
                    conn.execute(text(insert_final_metero_table_sql))
                    conn.commit()
                    print("Metero drought data successfully inserted into final table.")
                else:
                    print("No data found in staging table. Skipping final insert.")

        except Exception as e:
            print(f"Error inserting data into meterological_droughts: {e}")

    except Exception as e:
        print(f"General error in load_metero_data_to_database: {e}")
    # finally:
    #     print("Finalizing load_metero_data_to_database")

def load_metero_droughts_data():
    try:
        # Clear final table before inserting new records
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meterological_droughts"))
            print("Deleted existing records from meterological_droughts")

        # GCS bucket and directory (prefix)
        bucket_name = "datafiles_bucket"
        prefix = "downloads_metero/"

        # List files in GCS folder
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        print(f"Count of files found in GCS folder '{prefix}': {len(blobs)}")

        # Process each CSV file
        for blob in blobs:
            if blob.name.endswith(".csv"):
                #print("Processing file:", blob.name)

                # Download file to /tmp/ (Cloud Functions temporary directory)
                temp_path = f"/tmp/{os.path.basename(blob.name)}"
                blob.download_to_filename(temp_path)
                #print(f"Downloaded {blob.name} to {temp_path}")

                # Load data into database
                load_metero_data_to_database(temp_path)

                # Remove file to free space
                os.remove(temp_path)
                #print(f"Removed temp file {temp_path}")

    except Exception as e:
        print(f"Error in load_metero_droughts_data: {e}")
        
def load_indicator_categories(df_ind, engine):
    try:
        all_records = []

        # Explicit mapping for sheet → category type
        category_map = {
            "Poverty": "P",
            "Environment": "E",
            "Infrastructure": "I",
            "Health": "H"
        }

        for sheet_name, df in df_ind.items():
            # print (df.columns.tolist())
            df = normalize_columns(df)
            # print (df.columns.tolist())
            if 'indicator' not in df.columns:
                print(f"Skipping {sheet_name} — 'Indicator' column not found")
                continue

            # Use mapping (default to first char if sheet not in dict)
            category_type = category_map.get(sheet_name, sheet_name[0].upper())

            # Collect indicator records
            records = [
                (str(ind).strip(), category_type)
                for ind in df['indicator'].dropna()
            ]

            print(f"{sheet_name}: Found {len(records)} records")
            all_records.extend(records)

        # Deduplicate
        df_all_records = (
            pd.DataFrame(all_records, columns=['category_name', 'category_type'])
            .drop_duplicates()
        )
        print(f"Total unique records to insert: {len(df_all_records)}")

        if not df_all_records.empty:
            conn = engine.raw_connection()
            try:
                cur = conn.cursor()
                insert_query = """
                    INSERT INTO stg_indicator_categories (category_name, category_type)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """
                execute_values(cur, insert_query, df_all_records.to_records(index=False).tolist())
                conn.commit()

                #Insert data into the actual table  ON CONFLICT (category_name) DO NOTHING
                insert_query_actual = """
                    INSERT INTO indicator_categories (category_name, category_type)
                    SELECT category_name, category_type
                    FROM stg_indicator_categories
                """
                cur.execute(insert_query_actual)
                conn.commit()

                cur.close()
            except Exception as e:
                print(f"Error inserting indicator categories: {e}")
                conn.rollback()
                print("Transaction rolled back")
                return
            
            finally:
                conn.close()
        else:
            print("No records to insert")

    except Exception as e:
        print(f"Error inserting indicator categories: {e}")

def get_category_id(indicator_from_excel, cur):
    if pd.isna(indicator_from_excel):
        return None
    cur.execute("""
        SELECT category_id FROM indicator_categories
        WHERE LOWER(TRIM(category_name)) = LOWER(TRIM(%s))
        LIMIT 1;
    """, (indicator_from_excel.strip(),))
    result = cur.fetchone()
    return result[0] if result else None

def load_indicators(excel_path,sheet_csv_map, engine):
    cur = engine.raw_connection().cursor()
    # Loop through each sheet
    for sheet_name, csv_url in sheet_csv_map.items():
        print(f"\n📄 Processing sheet: {sheet_name}")

        try:
            excel_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            excel_df.columns = excel_df.columns.str.strip()
            excel_df['Indicator'] = excel_df['Indicator'].ffill()

            csv_df = pd.read_csv(csv_url, skiprows=[1])
            csv_df.columns = csv_df.columns.str.strip()

            # Match Variable from Excel with Indicator Name from CSV
            matched_df = pd.merge(
                csv_df,
                excel_df,
                left_on='Indicator Name',
                right_on='Variable',
                how='inner'
            )

            # Insert matched indicators in to staging table
            # Collect records first
            records = []

            for _, row in matched_df.iterrows():
                indicator_code = row['Indicator Code']
                indicator_name = row['Indicator Name']
                indicator_year = row['Year']
                value_index = row['Value']
                indicator_from_excel = row['Indicator']

                category_id = get_category_id(indicator_from_excel, cur)

                if category_id:
                    # truncate indicator_name to 100 chars if your DB column has VARCHAR(100)
                    records.append((
                        indicator_code,
                        indicator_name[:255],
                        indicator_year,
                        value_index,
                        category_id
                    ))
                else:
                    print(f"No category_id found for Indicator: {indicator_from_excel}")

            # Insert all records in one go 
            if records:
                try:
                    insert_query = """
                        INSERT INTO stg_indicators (
                            indicator_code, indicator_name, indicator_year, value_index, category_id
                        ) VALUES %s
                        ON CONFLICT DO NOTHING;
                    """
                    execute_values(cur, insert_query, records)
                    cur.connection.commit()
                    print(f"Inserted {len(records)} records into stg_indicators")

                    #Insert data into the actual table  ON CONFLICT (category_name) DO NOTHING
                    insert_query_actual = """
                            INSERT INTO indicators (indicator_code, indicator_name, indicator_year, value_index, category_id)
                            SELECT indicator_code, indicator_name, indicator_year, value_index, category_id
                            FROM stg_indicators
                            ON CONFLICT (indicator_year, indicator_name) DO NOTHING;
                    """
                    cur.execute(insert_query_actual)
                    cur.connection.commit()
                    print("Inserted records into indicators")

                except Exception as e:
                    print(f"Error inserting indicators: {e}")
            else:
                print("No valid records to insert.")

        except Exception as e:
            print(f"Failed to process sheet {sheet_name}: {e}")

def load_indicator_categories_data():
    #ind_datafile = "datafiles\CLEAN DATA - Humanitarian Data Exchange.xlsx"
    ind_datafile = "gs://datafiles_bucket/CLEAN DATA - Humanitarian Data Exchange.xlsx"
    sheet_csv_map = {
        'Poverty': 'https://data.humdata.org/dataset/57cd47b1-a017-4ecf-b175-d728659a2f03/resource/5d696f49-e7d4-4411-94b5-7b580bef0f01/download/poverty_arg.csv',
        'Environment': 'https://data.humdata.org/dataset/248aff86-89a1-4644-836b-7c4ca353c481/resource/74217609-e88e-40e8-ae8c-8a13cd61fc02/download/environment_arg.csv',
        'Infrastructure': 'https://data.humdata.org/dataset/50f557c8-1d4a-48b2-bf5a-d8b795484c59/resource/e89921d7-bd0d-4550-b4f6-fe7c9ed0ff87/download/infrastructure_arg.csv',
        'Health': 'https://data.humdata.org/dataset/507ff7ab-3efb-4216-8a71-d0f9cbdbe79b/resource/f2f8be4a-49b0-4f75-9d90-f8ec60375b8a/download/health_arg.csv'
    }

    try:
        df_ind = pd.read_excel(ind_datafile,sheet_name=None) # Read all sheets into a dictionary of DataFrames
        #Create staging tables categories and indicators
        create_table_sql_categories = """
        DROP TABLE IF EXISTS stg_indicator_categories;  
        CREATE TABLE IF NOT EXISTS stg_indicator_categories (
            category_name 	VARCHAR(255) PRIMARY KEY,
            category_type 	VARCHAR(20) NOT NULL
        );
        """
        create_table_sql_indicators = """
        DROP TABLE IF EXISTS stg_indicators;  
        CREATE TABLE IF NOT EXISTS stg_indicators (
                indicator_code	VARCHAR(30),
                indicator_name	VARCHAR(255) NOT NULL,
                indicator_year	NUMERIC,
                value_index		NUMERIC,
                category_id		SERIAL
        );
        """
        try:
            with engine.begin() as conn:
                conn.execute(text(create_table_sql_categories))
                conn.execute(text(create_table_sql_indicators))
                conn.commit()
        except Exception as e:
            print(f"Error creating staging table: {e}")
            return
        
        # Load data in indicator_categories 
        load_indicator_categories(df_ind, engine)
        # Load data in indicators
        load_indicators(ind_datafile,sheet_csv_map, engine)

    except Exception as e:
        print(f"Error reading indicators and indicator categories Excel file: {e}")
        return
    finally:
        print("Finalizing load_indicator_categories_data")

def insert_hydro_droughts_forecast():
    try:
        # Fetch SPI thresholds
        spi_query = "SELECT min_value, max_value, category FROM spi_thresholds"
        spi_thresholds = pd.read_sql(spi_query, engine)

        # Fetch hydrological drought data
        hydro_query = """
            SELECT h.station_code, s.station_name, h.daily_date, h.value_index
            FROM hydrological_droughts h
            JOIN stations s ON h.station_code = s.station_code
            ORDER BY s.station_name, h.daily_date
        """
        hydro_df = pd.read_sql(hydro_query, engine)
        hydro_df['daily_date'] = pd.to_datetime(hydro_df['daily_date'], format='%d-%m-%Y')

        stations = hydro_df['station_code'].unique()
        inserts = []

        for code in stations:
            city_df = hydro_df[hydro_df['station_code'] == code].sort_values('daily_date')

            if len(city_df) < 14:
                continue

            ts = city_df.set_index('daily_date')['value_index'].astype(float)

            # Create lag features
            lag = 7
            df_feat = pd.DataFrame()
            for i in range(1, lag + 1):
                df_feat[f'lag_{i}'] = ts.shift(i)
            df_feat['target'] = ts.values
            df_feat.dropna(inplace=True)

            X = df_feat.drop('target', axis=1).values
            y = df_feat['target'].values

            # Train XGBoost model
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                verbosity=0,
                random_state=42
            )
            model.fit(X, y)

            # Forecast next 5 days
            forecast_days = 5
            last_values = ts.values[-lag:].tolist()
            forecasts = []
            for _ in range(forecast_days):
                x_input = np.array(last_values[-lag:]).reshape(1, -1)
                pred = model.predict(x_input)[0]
                forecasts.append(pred)
                last_values.append(pred)

            last_date = ts.index[-1]
            forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]

            for forecast_date, forecast_value in zip(forecast_dates, forecasts):
                row = spi_thresholds[
                    (spi_thresholds['min_value'] <= forecast_value) &
                    (forecast_value < spi_thresholds['max_value'])
                ]
                risk_level = row.iloc[0]['category'] if not row.empty else None

                inserts.append({
                    'daily_date': forecast_date.strftime('%d-%m-%Y'),
                    'station_code': code,
                    'value_index': forecast_value,
                    'risk_level': risk_level,
                    'measurement_type': 'F',
                    'created_by': 'system',
                    'created_at': datetime.now()
                })

        if inserts:
            result_df = pd.DataFrame(inserts)
            insert_query = """
                INSERT INTO hydrological_droughts (
                    daily_date, station_code, value_index, risk_level, measurement_type, created_by, created_at
                )
                VALUES (:daily_date, :station_code, :value_index, :risk_level, :measurement_type, :created_by, :created_at)
                ON CONFLICT (daily_date, station_code) DO UPDATE
                SET value_index = EXCLUDED.value_index,
                    risk_level = EXCLUDED.risk_level,
                    measurement_type = EXCLUDED.measurement_type,
                    created_at = EXCLUDED.created_at;
            """
            with engine.begin() as conn:
                for _, row in result_df.iterrows():
                    conn.execute(text(insert_query), {
                        'daily_date': row['daily_date'],
                        'station_code': row['station_code'],
                        'value_index': row['value_index'],
                        'risk_level': row['risk_level'],
                        'measurement_type': row['measurement_type'],
                        'created_by': row['created_by'],
                        'created_at': row['created_at']
                    })
            print(f"Inserted {len(result_df)} forecast records (8 per station).")
        else:
            print("No forecasts inserted (insufficient history for all stations).")

    except Exception as e:
        print(f"Error during hydrological drought forecast insertion: {e}")

def insert_metero_droughts_forecast():
    try:
        try:
            query = "SELECT monthly_date, station_code, value_index FROM meterological_droughts WHERE measurement_type = 'A'"
            df = pd.read_sql(query, engine)
            df['monthly_date'] = pd.to_datetime(df['monthly_date'])
            df = df.sort_values(['station_code', 'monthly_date'])
        except Exception as e:
            print(f"Error fetching historical drought data: {e}")
            return

        # --- Fetch SPI Thresholds ---
        try:
            thresholds_df = pd.read_sql("SELECT category, min_value, max_value FROM spi_thresholds", engine)
        except Exception as e:
            print(f"Error fetching SPI thresholds: {e}")
            return

        # --- Forecasting Setup ---
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'ExtraTrees': ExtraTreesRegressor(random_state=42)
        }

        forecast_records = []
        # --- Loop Through Each Station ---
        for station in df['station_code'].unique():
            try:
                station_df = df[df['station_code'] == station].copy()
                station_df = station_df.set_index('monthly_date').sort_index()

                # --- Create Lag Features (2 lags) ---
                for lag in range(1, 3):
                    station_df[f'lag_{lag}'] = station_df['value_index'].shift(lag)

                station_df.dropna(inplace=True)

                if len(station_df) < 10:
                    continue

                # --- Chronological Train-Test Split (95/5) ---
                split_idx = int(len(station_df) * 0.95)
                train = station_df.iloc[:split_idx]
                test = station_df.iloc[split_idx:]

                X_train = train[[f'lag_{i}' for i in range(1, 3)]]
                y_train = train['value_index']
                X_test = test[[f'lag_{i}' for i in range(1, 3)]]
                y_test = test['value_index']

                # --- SMAPE Function ---
                def smape(y_true, y_pred):
                    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

                # --- Select Best Model by SMAPE ---
                best_model = None
                best_smape = float('inf')

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = smape(y_test.values, preds)
                    if score < best_smape:
                        best_smape = score
                        best_model = model

                # --- Forecast Next 3 Months ---
                last_known = station_df.iloc[-2:].copy()
                future_dates = [station_df.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]

                for date in future_dates:
                    lag_1 = last_known.iloc[-1]['value_index']
                    lag_2 = last_known.iloc[-2]['value_index']
                    X_future = np.array([[lag_1, lag_2]])
                    forecast = round(best_model.predict(X_future)[0], 3)

                    # --- Determine Risk Category ---
                    category = None
                    for _, row in thresholds_df.iterrows():
                        if row['min_value'] <= forecast <= row['max_value']:
                            category = row['category']
                            break

                    forecast_records.append({
                        'monthly_date': date,
                        'station_code': station,
                        'value_index': forecast,
                        'measurement_type': 'F',
                        'risk_level': category
                    })

                    # --- Update Lag History ---
                    last_known = pd.concat([
                        last_known,
                        pd.DataFrame({'value_index': [forecast]}, index=[date])
                    ]).iloc[-2:]

            except Exception as e:
                print(f"Error processing station {station}: {e}")
                continue

        # --- Insert Forecasts into DB ---
        if forecast_records:
            insert_query = text("""
                INSERT INTO meterological_droughts (
                    monthly_date, station_code, value_index, measurement_type, risk_level, created_at
                )
                VALUES (:monthly_date, :station_code, :value_index, :measurement_type, :risk_level, :created_at)
                ON CONFLICT (monthly_date, station_code) DO UPDATE
                SET value_index = EXCLUDED.value_index,
                    measurement_type = EXCLUDED.measurement_type,
                    risk_level = EXCLUDED.risk_level,
                    created_at = EXCLUDED.created_at;
            """)
            try:
                with engine.begin() as conn:
                    for record in forecast_records:
                        conn.execute(insert_query, {
                            "monthly_date": record['monthly_date'],
                            "station_code": record['station_code'],
                            "value_index": float(record['value_index']),
                            "measurement_type": record['measurement_type'],
                            "risk_level": record['risk_level'],
                            "created_at": datetime.now()
                        })
                print("Forecasts inserted into meterological_droughts")
            except Exception as e:
                print(f"Error inserting forecast records: {e}")
        else:
            print("No forecasts generated.")

    except Exception as e:
        print(f"Fatal error in insert_metero_droughts_forecast: {e}")

try:
    # db_url = f"postgresql://postgres:Database%40123@34.100.141.55:5432/HO_IFRC_ARG"
    db_url = f"postgresql://postgres:Database%40123@34.100.141.55:5432/ho_ifrc_arg"
    engine = create_engine(db_url)
except Exception as e:
    print(f"Error creating database engine: {e}")