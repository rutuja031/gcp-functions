## cloud function for daily temp insert and forecast

# this code now works because cloud run internally uses htttp trigger this code fails 

# requirements.txt:
# pandas
# requests
# rapidfuzz
# numpy
# sqlalchemy
# psycopg2-binary
# scikit-learn
# xgboost

# DB_CONFIG = {
#         'dbname': "HO_IFRC_ARG",
#         'user': 'postgres',
#         'password': "Database@123",
#         'host': "34.100.141.55",
#         'port': 5432
#     }
 
 
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
warnings.filterwarnings("ignore", category=UserWarning)

# DATABASE CONNECTION
DB_USER = "postgres"
DB_PASS = "Database@123"
DB_HOST = "34.100.141.55"
DB_PORT = "5432"
DB_NAME = "HO_IFRC_ARG"

DB_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# DB_URL = (
#     f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASS']}"
#     f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
# )
engine = create_engine(DB_URL)

# STEP 1: FETCH DAILY TEMPERATURE DATA
def insert_daily_temperature():
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


# STEP 2: FORECAST TEMPERATURE
def insert_forecast_temperature():
    try:
        forecast_days = 7
        lags = 7
        record_limit = 750

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
                            LIMIT {record_limit}
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

# ## entry point function for pub/sub job created with msg body:{"job": "daily"} 
def run_all_temp(event, context):
    """
    Cloud Function entry point triggered by Pub/Sub.
    `event` contains the Pub/Sub message.
    """
    try:
        print(f"Received Pub/Sub event: {event}")

        # Decode message data
        if "data" in event:
            message = base64.b64decode(event["data"]).decode("utf-8")
            print(f"Decoded message: {message}")
            try:
                payload = json.loads(message)
            except:
                payload = {"job": message}
        else:
            payload = {"job": "manual"}

        job_type = payload.get("job", "unknown")
        print(f"Job type: {job_type}")

        # Run your tasks
        insert_daily_temperature()
        insert_forecast_temperature()

        print("All tasks completed successfully.")

    except Exception as e:
        print(f"Error in run_all_temp: {str(e)}")
    
