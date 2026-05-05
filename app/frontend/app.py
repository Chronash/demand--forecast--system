import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "http://localhost:8000"
MODELS_DIR = Path("artifacts/models")
DATA_DIR = Path("artifacts/generated")

st.set_page_config(
    page_title="Прогноз спроса — Borjomi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_sales_data():
    df = pd.read_csv(DATA_DIR / "sales_daily.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


@st.cache_data
def load_metadata():
    with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def make_prediction(payload: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Не удаётся подключиться к API. "
            "Сначала запусти FastAPI: python -m uvicorn app.api.main:app --port 8000"
        )
        return None
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")
        return None


st.sidebar.title("💧 Параметры прогноза")
st.sidebar.markdown("---")

region = st.sidebar.selectbox(
    "Регион",
    ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"],
    index=0,
)

store_type = st.sidebar.selectbox(
    "Тип магазина",
    ["Супермаркет", "Минимаркет", "Гипермаркет", "АЗС"],
    index=0,
)

store_id = st.sidebar.slider("ID магазина", 1, 400, 50)
month = st.sidebar.slider("Месяц", 1, 12, 7)
weekday = st.sidebar.slider("День недели (0=Пн, 6=Вс)", 0, 6, 5)
temperature = st.sidebar.slider("Температура (°C)", -20.0, 45.0, 24.0, step=0.5)
price = st.sidebar.slider("Цена (₽)", 30.0, 120.0, 58.0, step=0.5)
is_weekend = st.sidebar.checkbox("Выходной день", value=weekday >= 5)
is_holiday = st.sidebar.checkbox("Праздничный день", value=False)
promo_flag = st.sidebar.checkbox("Промо-акция", value=False)

predict_btn = st.sidebar.button("🔮 Получить прогноз", width="stretch")

st.title("💧 Система прогнозирования спроса на минеральную воду")
st.markdown(
    "Интеллектуальная система на основе XGBoost для прогнозирования "
    "ежедневного спроса в зависимости от региона, цены, температуры, "
    "промо-активности и сезонности."
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Прогноз", "📊 Аналитика", "🌡️ Сценарии", "📋 О модели"]
)

day_of_year = int(pd.Timestamp(year=2024, month=month, day=15).dayofyear)

base_payload = {
    "store_id": store_id,
    "region": region,
    "store_type": store_type,
    "month": month,
    "weekday": weekday,
    "day_of_year": day_of_year,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday,
    "promo_flag": promo_flag,
    "temperature": float(temperature),
    "price": float(price),
}

with tab1:
    st.subheader("Прогноз спроса")

    if predict_btn:
        with st.spinner("Выполняется прогноз..."):
            result = make_prediction(base_payload)

        if result:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                label="📦 Прогноз спроса",
                value=f"{result['predicted_demand']:,} шт.",
            )
            col2.metric(
                label="💰 Ожидаемая выручка",
                value=f"{result['predicted_revenue']:,.0f} ₽",
            )
            col3.metric(
                label="🤖 Модель",
                value=result["model_name"].upper(),
                help=f"Версия: {result['model_version']}",
            )

            st.success("✅ Прогноз успешно получен!")
    else:
        st.info("👈 Настрой параметры слева и нажми «Получить прогноз».")

with tab2:
    st.subheader("Аналитика по данным")

    with st.spinner("Загрузка данных..."):
        df = load_sales_data()

    col1, col2 = st.columns(2)

    with col1:
        monthly = df.groupby("month")["demand_units"].mean().reset_index()
        monthly.columns = ["Месяц", "Средний спрос"]

        fig_season = px.line(
            monthly,
            x="Месяц",
            y="Средний спрос",
            markers=True,
            title="Сезонность: средний спрос по месяцам",
        )
        fig_season.update_traces(line_color="#2196F3", line_width=2.5)
        st.plotly_chart(fig_season, width="stretch")

    with col2:
        region_data = (
            df.groupby("region")["demand_units"]
            .mean()
            .reset_index()
            .sort_values("demand_units", ascending=False)
        )
        region_data.columns = ["Регион", "Средний спрос"]

        fig_region = px.bar(
            region_data,
            x="Регион",
            y="Средний спрос",
            title="Средний спрос по регионам",
            color="Средний спрос",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_region, width="stretch")

    sample = df.sample(min(8000, len(df)), random_state=42)

    fig_temp = px.scatter(
        sample,
        x="temperature",
        y="demand_units",
        opacity=0.35,
        title="Влияние температуры на спрос",
        labels={"temperature": "Температура (°C)", "demand_units": "Спрос (шт.)"},
        trendline="ols",
        trendline_color_override="#F44336",
    )
    st.plotly_chart(fig_temp, width="stretch")

with tab3:
    st.subheader("Сценарный анализ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Влияние цены на спрос")
        prices = list(range(40, 101, 5))
        demand_by_price = []

        for p in prices:
            payload = {**base_payload, "price": float(p)}
            r = make_prediction(payload)
            if r:
                demand_by_price.append(
                    {"Цена (₽)": p, "Спрос (шт.)": r["predicted_demand"]}
                )

        if demand_by_price:
            df_price = pd.DataFrame(demand_by_price)
            fig_price = px.line(
                df_price,
                x="Цена (₽)",
                y="Спрос (шт.)",
                markers=True,
                title="Как меняется спрос при изменении цены",
            )
            st.plotly_chart(fig_price, width="stretch")

    with col2:
        st.markdown("#### Влияние температуры на спрос")
        temps = list(range(-10, 41, 5))
        demand_by_temp = []

        for t in temps:
            payload = {**base_payload, "temperature": float(t)}
            r = make_prediction(payload)
            if r:
                demand_by_temp.append(
                    {"Температура (°C)": t, "Спрос (шт.)": r["predicted_demand"]}
                )

        if demand_by_temp:
            df_temp = pd.DataFrame(demand_by_temp)
            fig_t = px.line(
                df_temp,
                x="Температура (°C)",
                y="Спрос (шт.)",
                markers=True,
                title="Как меняется спрос при изменении температуры",
            )
            st.plotly_chart(fig_t, width="stretch")

    st.markdown("#### Сравнение регионов")
    all_regions = ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"]
    region_results = []

    for reg in all_regions:
        payload = {**base_payload, "region": reg}
        r = make_prediction(payload)
        if r:
            region_results.append({"Регион": reg, "Спрос (шт.)": r["predicted_demand"]})

    if region_results:
        df_regions = pd.DataFrame(region_results).sort_values(
            "Спрос (шт.)", ascending=False
        )
        fig_regions = px.bar(
            df_regions,
            x="Регион",
            y="Спрос (шт.)",
            title="Прогноз спроса по регионам при текущих параметрах",
            color="Спрос (шт.)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_regions, width="stretch")

with tab4:
    st.subheader("Информация о модели")

    try:
        meta = load_metadata()

        c1, c2 = st.columns(2)
        c1.markdown(f"**Лучшая модель:** `{meta['best_model']}`")
        c2.markdown(f"**Версия модели:** `{meta['version']}`")

        st.markdown("#### Метрики моделей")
        metrics_rows = []
        for model_name, m in meta["metrics"].items():
            metrics_rows.append(
                {
                    "Модель": model_name,
                    "MAE": m["mae"],
                    "RMSE": m["rmse"],
                    "R²": m["r2"],
                }
            )

        st.dataframe(pd.DataFrame(metrics_rows), width="stretch")

        st.markdown("#### Используемые признаки")
        st.code("\n".join(meta["feature_columns"]))

        st.markdown("#### Доступные регионы")
        st.write(", ".join(meta["regions"]))

        st.markdown("#### Типы магазинов")
        st.write(", ".join(meta["store_types"]))

    except FileNotFoundError:
        st.warning("Не найден metadata.json. Сначала выполни `python -m ml.train`.")