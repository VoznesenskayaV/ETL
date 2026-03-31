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
    description="Variant 01: Moscow weather ETL + ML + chart",
    schedule_interval=None,
    catchup=False,
)


def fetch_weather_forecast():
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

    output_path = os.path.join(data_dir, "variant_01_raw.csv")
    df.to_csv(output_path, index=False)

    print(f"Погодные данные сохранены: {output_path}")
    print(df)


def clean_weather_data():
    import pandas as pd

    data_dir = "/opt/airflow/data"
    input_path = os.path.join(data_dir, "variant_01_raw.csv")
    output_path = os.path.join(data_dir, "variant_01_result.csv")

    df = pd.read_csv(input_path)

    df = df[["date", "avgtemp_c"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["avgtemp_c"] = df["avgtemp_c"].ffill()

    df.to_csv(output_path, index=False)

    print(f"Очищенные погодные данные сохранены: {output_path}")
    print(df)


def fetch_sales_data():
    import pandas as pd

    data_dir = "/opt/airflow/data"
    weather_path = os.path.join(data_dir, "variant_01_result.csv")
    sales_path = os.path.join(data_dir, "variant_01_sales.csv")

    weather_df = pd.read_csv(weather_path)
    dates = weather_df["date"].tolist()

    # Учебные данные продаж для демонстрации ML
    base_sales = [120, 150, 135]
    sales = base_sales[:len(dates)]

    df = pd.DataFrame({
        "date": dates,
        "sales": sales
    })

    df.to_csv(sales_path, index=False)

    print(f"Данные продаж сохранены: {sales_path}")
    print(df)


def clean_sales_data():
    import pandas as pd

    data_dir = "/opt/airflow/data"
    input_path = os.path.join(data_dir, "variant_01_sales.csv")
    output_path = os.path.join(data_dir, "variant_01_clean_sales.csv")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["sales"] = df["sales"].ffill()

    df.to_csv(output_path, index=False)

    print(f"Очищенные данные продаж сохранены: {output_path}")
    print(df)


def join_datasets():
    import pandas as pd

    data_dir = "/opt/airflow/data"
    weather_path = os.path.join(data_dir, "variant_01_result.csv")
    sales_path = os.path.join(data_dir, "variant_01_clean_sales.csv")
    output_path = os.path.join(data_dir, "variant_01_joined.csv")

    weather_df = pd.read_csv(weather_path)
    sales_df = pd.read_csv(sales_path)

    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.strftime("%Y-%m-%d")
    sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.strftime("%Y-%m-%d")

    joined_df = pd.merge(weather_df, sales_df, on="date", how="inner")
    joined_df.to_csv(output_path, index=False)

    print(f"Объединённый датасет сохранён: {output_path}")
    print(joined_df)


def train_ml_model():
    import pandas as pd
    import joblib
    from sklearn.linear_model import LinearRegression

    data_dir = "/opt/airflow/data"
    input_path = os.path.join(data_dir, "variant_01_joined.csv")
    model_path = os.path.join(data_dir, "variant_01_ml_model.pkl")
    pred_path = os.path.join(data_dir, "variant_01_predictions.csv")

    df = pd.read_csv(input_path)

    X = df[["avgtemp_c"]].rename(columns={"avgtemp_c": "temperature"})
    y = df["sales"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)

    df["predicted_sales"] = model.predict(X)
    df.to_csv(pred_path, index=False)

    print(f"ML-модель сохранена: {model_path}")
    print(f"Прогнозы сохранены: {pred_path}")
    print(df)


def plot_bar_chart():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = "/opt/airflow/data"
    input_path = os.path.join(data_dir, "variant_01_predictions.csv")
    plot_path = os.path.join(data_dir, "variant_01_plot.png")

    df = pd.read_csv(input_path)
    df = df.sort_values(by="predicted_sales")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df["date"], df["predicted_sales"])

    plt.title("Прогнозируемые продажи по дням")
    plt.xlabel("Дата")
    plt.ylabel("Прогноз продаж")
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"График сохранён: {plot_path}")


t1 = PythonOperator(
    task_id="fetch_weather_forecast",
    python_callable=fetch_weather_forecast,
    dag=dag,
)

t2 = PythonOperator(
    task_id="clean_weather_data",
    python_callable=clean_weather_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id="fetch_sales_data",
    python_callable=fetch_sales_data,
    dag=dag,
)

t4 = PythonOperator(
    task_id="clean_sales_data",
    python_callable=clean_sales_data,
    dag=dag,
)

t5 = PythonOperator(
    task_id="join_datasets",
    python_callable=join_datasets,
    dag=dag,
)

t6 = PythonOperator(
    task_id="train_ml_model",
    python_callable=train_ml_model,
    dag=dag,
)

t7 = PythonOperator(
    task_id="plot_bar_chart",
    python_callable=plot_bar_chart,
    dag=dag,
)

t1 >> t2
t3 >> t4
[t2, t4] >> t5
t5 >> t6 >> t7
