import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

dag = DAG(
    dag_id="variant_01_moscow_3days",
    default_args=default_args,
    description="Variant 01: Moscow weather forecast for 3 days, keep date and avgtemp_c, save to CSV.",
    schedule_interval=None,
    catchup=False,
)

def fetch_weather_data():
    import requests
    import pandas as pd

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=55.75&longitude=37.62"
        "&daily=temperature_2m_mean"
        "&timezone=Europe%2FMoscow"
        "&forecast_days=3"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    dates = data["daily"]["time"]
    temperatures = data["daily"]["temperature_2m_mean"]

    df = pd.DataFrame({
        "date": dates,
        "avgtemp_c": temperatures
    })

    data_dir = "/opt/airflow/data"
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "variant_01_raw.csv"), index=False)

    print("Weather data saved")

def transform_weather_data():
    import pandas as pd

    data_dir = "/opt/airflow/data"
    df = pd.read_csv(os.path.join(data_dir, "variant_01_raw.csv"))

    df = df[["date", "avgtemp_c"]]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["avgtemp_c"] = df["avgtemp_c"].ffill()

    df.to_csv(os.path.join(data_dir, "variant_01_result.csv"), index=False)

    print(df)

t1 = PythonOperator(
    task_id="fetch_weather_data",
    python_callable=fetch_weather_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id="transform_weather_data",
    python_callable=transform_weather_data,
    dag=dag,
)

t1 >> t2
