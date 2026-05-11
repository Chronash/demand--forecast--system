import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit as st

warnings.filterwarnings("ignore")

API_URL = "http://localhost:8000"
MODELS_DIR = Path("artifacts/models")
DATA_DIR = Path("artifacts/generated")
HISTORY_DIR = Path("artifacts/history")
HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

REGIONS = ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"]
STORE_TYPES = ["Супермаркет", "Минимаркет", "Гипермаркет", "АЗС"]

st.set_page_config(
    page_title="Borjomi Demand Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: rgba(12, 19, 32, 0.72);
        --panel-strong: rgba(11, 18, 31, 0.84);
        --panel-border: rgba(148, 163, 184, 0.10);
        --text: #eef4ff;
        --muted: #9fb3c8;
        --soft: #7f93aa;
        --accent: #3ecf8e;
        --accent-2: #38bdf8;
        --accent-3: #8b5cf6;
        --danger: #fb7185;
        --warning: #fbbf24;
        --shadow-soft: 0 12px 32px rgba(0, 0, 0, 0.18);
        --shadow-card: 0 10px 26px rgba(0, 0, 0, 0.16);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.08), transparent 24%),
            linear-gradient(180deg, #06101d 0%, #081220 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 2.2rem;
        max-width: 1520px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(8, 15, 27, 0.88) 0%, rgba(9, 18, 31, 0.88) 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--muted) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 1.2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(139, 92, 246, 0.08) 42%, rgba(62, 207, 142, 0.07)),
            rgba(12, 19, 32, 0.70);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    .hero-card:before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.06), transparent 28%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.9rem;
        font-weight: 800;
        line-height: 1.04;
        color: white;
        margin: 0 0 10px 0;
    }

    .hero-subtitle {
        max-width: 980px;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.75;
        margin: 0;
    }

    .section-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(12, 19, 32, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.10);
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(8px);
        margin-bottom: 0.9rem;
    }

    .tiny-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-2);
        margin-bottom: 7px;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.62;
    }

    .status-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(12, 19, 32, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.10);
        box-shadow: var(--shadow-card);
        margin-bottom: 1rem;
    }

    .status-label {
        color: var(--soft);
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 8px;
    }

    .status-value {
        color: white;
        font-size: 1.22rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .status-help {
        color: var(--muted);
        font-size: 0.87rem;
        margin-top: 6px;
    }

    div[data-testid="metric-container"] {
        background: rgba(12, 19, 32, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.10);
        padding: 18px 18px 16px 18px;
        border-radius: 20px;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(9, 16, 28, 0.46);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 16px;
        padding: 6px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        color: var(--muted);
        border-radius: 12px;
        padding: 0 16px;
        font-weight: 700;
        font-size: 0.97rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(62, 207, 142, 0.12));
        color: white !important;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, rgba(62, 207, 142, 0.94), rgba(56, 189, 248, 0.94));
        color: #04111e;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 10px 24px rgba(56, 189, 248, 0.14);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        filter: brightness(1.04);
        transform: translateY(-1px);
        transition: 0.2s ease;
    }

    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(148,163,184,0.12) !important;
        color: white !important;
    }

    .stDataFrame, .stTable {
        border-radius: 18px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        background: rgba(15, 23, 42, 0.70) !important;
        color: white !important;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148,163,184,0.20), transparent);
        margin: 1.1rem 0;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.2rem;
    }

    .sidebar-sub {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }

    .mini-note {
        margin-top: 8px;
        color: var(--soft);
        font-size: 0.8rem;
        line-height: 1.45;
    }

    .glass-note {
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,163,184,0.08);
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 0.75rem;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.05rem;
        }

        .hero-subtitle {
            font-size: 0.98rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_SESSION_STATE = {
    "last_prediction_payload": None,
    "last_prediction_result": None,
    "scenario_payload": None,
    "scenario_price_df": None,
    "scenario_temp_df": None,
    "scenario_region_df": None,
    "scenario_a_payload": None,
    "scenario_a_result": None,
    "scenario_b_payload": None,
    "scenario_b_result": None,
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def payload_to_key(payload: dict):
    return tuple(sorted(payload.items()))


def scenario_payload_to_key(payload: dict):
    return tuple(sorted(payload.items()))


@st.cache_resource
def get_api_session():
    return requests.Session()


@st.cache_data
def load_sales_data():
    path = DATA_DIR / "sales_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


@st.cache_data
def load_metadata():
    with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_cv_results():
    path = MODELS_DIR / "cv_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def ensure_history_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_prediction_to_history(payload: dict, result: dict):
    ensure_history_dir()

    row = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "store_id": payload["store_id"],
        "region": payload["region"],
        "store_type": payload["store_type"],
        "month": payload["month"],
        "weekday": payload["weekday"],
        "day_of_year": payload["day_of_year"],
        "is_weekend": payload["is_weekend"],
        "is_holiday": payload["is_holiday"],
        "promo_flag": payload["promo_flag"],
        "temperature": payload["temperature"],
        "price": payload["price"],
        "predicted_demand": result["predicted_demand"],
        "predicted_revenue": result["predicted_revenue"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
    }

    row_df = pd.DataFrame([row])
    file_exists = HISTORY_FILE.exists()

    row_df.to_csv(
        HISTORY_FILE,
        mode="a",
        index=False,
        header=not file_exists,
        encoding="utf-8-sig",
    )


def load_prediction_history():
    ensure_history_dir()

    if HISTORY_FILE.exists():
        df = pd.read_csv(HISTORY_FILE)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df

    return pd.DataFrame()


def clear_prediction_history():
    ensure_history_dir()
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


@st.cache_data
def compute_shap_values():
    import joblib
    from xgboost import XGBRegressor

    meta = load_metadata()
    feature_columns = meta["feature_columns"]

    df = pd.read_csv(DATA_DIR / "sales_daily.csv")
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df = df.sort_values("sale_date").reset_index(drop=True)

    region_encoder = joblib.load(MODELS_DIR / "label_encoder_region.pkl")
    store_type_encoder = joblib.load(MODELS_DIR / "label_encoder_store_type.pkl")

    if "region" in df.columns:
        df["region"] = region_encoder.transform(df["region"])
    if "store_type" in df.columns:
        df["store_type"] = store_type_encoder.transform(df["store_type"])

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"В датасете отсутствуют признаки: {missing_features}")

    X_df = df[feature_columns].copy()

    model = XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    sample_size = min(2000, len(X_df))
    X_sample = X_df.iloc[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, feature_columns


def make_prediction(payload: dict):
    session = get_api_session()

    try:
        response = session.post(f"{API_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Не удаётся подключиться к API. "
            "Сначала запусти FastAPI: python -m uvicorn app.api.main:app --port 8000 --reload"
        )
        return None
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")
        return None


def style_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter", color="#EAF2FF"),
        title=dict(font=dict(size=18, color="#FFFFFF")),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color="#DCE8F5"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
        ),
    )
    return fig


def save_named_scenario(name: str, payload: dict):
    result = make_prediction(payload)
    if not result:
        return

    if name == "A":
        st.session_state.scenario_a_payload = payload.copy()
        st.session_state.scenario_a_result = result
    elif name == "B":
        st.session_state.scenario_b_payload = payload.copy()
        st.session_state.scenario_b_result = result


def format_scenario_payload(payload: dict):
    return pd.DataFrame(
        [
            {
                "Регион": payload["region"],
                "Тип магазина": payload["store_type"],
                "ID магазина": payload["store_id"],
                "Месяц": payload["month"],
                "День недели": payload["weekday"],
                "Температура": payload["temperature"],
                "Цена": payload["price"],
                "Выходной": payload["is_weekend"],
                "Праздник": payload["is_holiday"],
                "Промо": payload["promo_flag"],
            }
        ]
    )


def compare_scenarios(result_a: dict, result_b: dict):
    demand_a = result_a["predicted_demand"]
    demand_b = result_b["predicted_demand"]

    revenue_a = result_a["predicted_revenue"]
    revenue_b = result_b["predicted_revenue"]

    demand_diff = demand_b - demand_a
    revenue_diff = revenue_b - revenue_a

    demand_pct = (demand_diff / demand_a * 100) if demand_a else 0
    revenue_pct = (revenue_diff / revenue_a * 100) if revenue_a else 0

    return {
        "demand_diff": demand_diff,
        "revenue_diff": revenue_diff,
        "demand_pct": demand_pct,
        "revenue_pct": revenue_pct,
    }


st.sidebar.markdown('<div class="sidebar-title">💧 Control Panel</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-sub">Настрой параметры спроса, цены, сезона и промо-активности для получения прогноза в реальном времени.</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div class="glass-note">Начни с базового сценария, затем меняй 1–2 параметра, чтобы наглядно увидеть влияние на спрос.</div>',
    unsafe_allow_html=True,
)

region = st.sidebar.selectbox("Регион", REGIONS, index=0)
store_type = st.sidebar.selectbox("Тип магазина", STORE_TYPES, index=0)

store_id = st.sidebar.slider("ID магазина", 1, 400, 50)
month = st.sidebar.slider("Месяц", 1, 12, 7)
weekday = st.sidebar.slider("День недели (0=Пн, 6=Вс)", 0, 6, 5)
temperature = st.sidebar.slider("Температура (°C)", -35.0, 35.0, 24.0, step=0.5)
price = st.sidebar.slider("Цена (₽)", 70.0, 140.0, 95.0, step=0.5)
is_weekend = st.sidebar.checkbox("Выходной день", value=weekday >= 5)
is_holiday = st.sidebar.checkbox("Праздничный день", value=False)
promo_flag = st.sidebar.checkbox("Промо-акция", value=False)

predict_btn = st.sidebar.button("⚡ Получить прогноз", width="stretch")

st.sidebar.markdown(
    '<div class="mini-note">Модель: XGBoost · Реалистичный synthetic retail dataset · Premium UI</div>',
    unsafe_allow_html=True,
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

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Интеллектуальная система прогнозирования спроса</div>
        <p class="hero-subtitle">
            Аналитический интерфейс для прогноза спроса на минеральную воду:
            оценивай влияние региона, температуры, цены и промо-активности,
            сравнивай сценарии и отслеживай историю прогнозов в едином dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Region</div>
            <div class="status-value">{region}</div>
            <div class="status-help">Активный регион сценария</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col2:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Store Type</div>
            <div class="status-value">{store_type}</div>
            <div class="status-help">Формат торговой точки</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col3:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Temperature</div>
            <div class="status-value">{temperature:.1f} °C</div>
            <div class="status-help">Параметр текущего сценария</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col4:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Price</div>
            <div class="status-value">{price:.1f} ₽</div>
            <div class="status-help">Текущая отпускная цена</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🔮 Прогноз",
        "📊 Аналитика",
        "🌡️ Сценарии и A/B",
        "🧠 Объяснение модели",
        "🗂️ История прогнозов",
        "📋 О модели",
    ]
)

with tab1:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Live Forecast</div>
            <div class="section-title">Текущий прогноз</div>
            <div class="section-subtitle">Ключевые показатели текущего сценария.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_payload_key = payload_to_key(base_payload)

    if predict_btn:
        result = None

        if st.session_state.last_prediction_payload == current_payload_key:
            result = st.session_state.last_prediction_result
        else:
            with st.spinner("Выполняется прогноз..."):
                result = make_prediction(base_payload)

            if result:
                st.session_state.last_prediction_payload = current_payload_key
                st.session_state.last_prediction_result = result
                save_prediction_to_history(base_payload, result)

        if result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Прогноз спроса", f"{result['predicted_demand']:,} шт.")
            col2.metric("Ожидаемая выручка", f"{result['predicted_revenue']:,.0f} ₽")
            col3.metric(
                "Модель",
                result["model_name"].upper(),
                help=f"Версия: {result['model_version']}",
            )
            st.success("Прогноз успешно рассчитан и сохранён в историю.")
    elif st.session_state.last_prediction_result is not None:
        result = st.session_state.last_prediction_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Прогноз спроса", f"{result['predicted_demand']:,} шт.")
        col2.metric("Ожидаемая выручка", f"{result['predicted_revenue']:,.0f} ₽")
        col3.metric(
            "Модель",
            result["model_name"].upper(),
            help=f"Версия: {result['model_version']}",
        )
        st.info("Показан последний рассчитанный прогноз.")
    else:
        st.info("Выбери параметры слева и нажми «Получить прогноз».")

    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Scenario Input</div>
            <div class="section-title">Текущая конфигурация</div>
            <div class="section-subtitle">Параметры, используемые для прогноза и сценарного анализа.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_df = pd.DataFrame([base_payload])
    st.dataframe(preview_df, width="stretch")
    st.caption("Текущие параметры используются для основного прогноза и всех сценарных расчётов.")

with tab2:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Dataset Analytics</div>
            <div class="section-title">Аналитика по историческим данным</div>
            <div class="section-subtitle">Сезонность, региональные различия и влияние температуры на спрос.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_sales_data()

    if df.empty:
        st.warning("Файл sales_daily.csv не найден. Сначала выполни `python -m ml.generate_dataset`.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            monthly = df.groupby("month")["demand_units"].mean().reset_index()
            monthly.columns = ["Месяц", "Средний спрос"]

            fig_season = px.line(
                monthly,
                x="Месяц",
                y="Средний спрос",
                markers=True,
                title="Сезонность спроса",
            )
            fig_season.update_traces(line_color="#38bdf8", line_width=3)
            style_figure(fig_season)
            st.plotly_chart(fig_season, width="stretch")
            st.caption("Помогает увидеть месяцы с наиболее высоким и низким средним спросом.")

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
                color_continuous_scale="Tealgrn",
            )
            style_figure(fig_region)
            st.plotly_chart(fig_region, width="stretch")
            st.caption("Показывает различия спроса между регионами на исторических данных.")

        sample = df.sample(min(8000, len(df)), random_state=42)

        fig_temp = px.scatter(
            sample,
            x="temperature",
            y="demand_units",
            opacity=0.35,
            title="Температура и спрос",
            labels={"temperature": "Температура (°C)", "demand_units": "Спрос (шт.)"},
            trendline="ols",
            trendline_color_override="#3ecf8e",
        )
        style_figure(fig_temp)
        st.plotly_chart(fig_temp, width="stretch")
        st.caption("Позволяет визуально оценить связь между температурой и объёмом продаж.")

with tab3:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">What-if Analysis</div>
            <div class="section-title">Сценарный анализ</div>
            <div class="section-subtitle">Как цена, температура и регион влияют на прогнозируемый спрос.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_scenario_key = scenario_payload_to_key(base_payload)

    if st.session_state.scenario_payload != current_scenario_key:
        with st.spinner("Пересчёт сценарного анализа..."):
            prices = list(range(70, 141, 10))
            demand_by_price = []

            for p in prices:
                payload = {**base_payload, "price": float(p)}
                r = make_prediction(payload)
                if r:
                    demand_by_price.append({"Цена (₽)": p, "Спрос (шт.)": r["predicted_demand"]})

            temps = list(range(-30, 36, 10))
            demand_by_temp = []

            for t in temps:
                payload = {**base_payload, "temperature": float(t)}
                r = make_prediction(payload)
                if r:
                    demand_by_temp.append({"Температура (°C)": t, "Спрос (шт.)": r["predicted_demand"]})

            region_results = []
            for reg in REGIONS:
                payload = {**base_payload, "region": reg}
                r = make_prediction(payload)
                if r:
                    region_results.append({"Регион": reg, "Спрос (шт.)": r["predicted_demand"]})

            st.session_state.scenario_price_df = pd.DataFrame(demand_by_price)
            st.session_state.scenario_temp_df = pd.DataFrame(demand_by_temp)
            st.session_state.scenario_region_df = pd.DataFrame(region_results)
            st.session_state.scenario_payload = current_scenario_key

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Влияние цены")
        if st.session_state.scenario_price_df is not None and not st.session_state.scenario_price_df.empty:
            fig_price = px.line(
                st.session_state.scenario_price_df,
                x="Цена (₽)",
                y="Спрос (шт.)",
                markers=True,
                title="Изменение спроса при изменении цены",
            )
            fig_price.update_traces(line_color="#38bdf8", line_width=3)
            style_figure(fig_price)
            st.plotly_chart(fig_price, width="stretch")

    with col2:
        st.markdown("#### Влияние температуры")
        if st.session_state.scenario_temp_df is not None and not st.session_state.scenario_temp_df.empty:
            fig_t = px.line(
                st.session_state.scenario_temp_df,
                x="Температура (°C)",
                y="Спрос (шт.)",
                markers=True,
                title="Изменение спроса при изменении температуры",
            )
            fig_t.update_traces(line_color="#3ecf8e", line_width=3)
            style_figure(fig_t)
            st.plotly_chart(fig_t, width="stretch")

    st.markdown("#### Сравнение регионов")
    if st.session_state.scenario_region_df is not None and not st.session_state.scenario_region_df.empty:
        df_regions = st.session_state.scenario_region_df.sort_values("Спрос (шт.)", ascending=False)
        fig_regions = px.bar(
            df_regions,
            x="Регион",
            y="Спрос (шт.)",
            title="Прогноз спроса по регионам",
            color="Спрос (шт.)",
            color_continuous_scale="Tealgrn",
        )
        style_figure(fig_regions)
        st.plotly_chart(fig_regions, width="stretch")

    st.caption("Сценарии автоматически обновляются при изменении входных параметров.")

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Scenario Comparison</div>
            <div class="section-title">Сравнение двух сценариев</div>
            <div class="section-subtitle">Сохрани текущую конфигурацию как сценарий A или B и сравни итоговый спрос и выручку.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

    with btn_col1:
        if st.button("💾 Сохранить как сценарий A", width="stretch"):
            with st.spinner("Сохраняем сценарий A..."):
                save_named_scenario("A", base_payload)
            if st.session_state.scenario_a_result:
                st.success("Сценарий A сохранён.")

    with btn_col2:
        if st.button("💾 Сохранить как сценарий B", width="stretch"):
            with st.spinner("Сохраняем сценарий B..."):
                save_named_scenario("B", base_payload)
            if st.session_state.scenario_b_result:
                st.success("Сценарий B сохранён.")

    with btn_col3:
        if st.button("🗑️ Очистить A/B", width="stretch"):
            st.session_state.scenario_a_payload = None
            st.session_state.scenario_a_result = None
            st.session_state.scenario_b_payload = None
            st.session_state.scenario_b_result = None
            st.rerun()

    scenario_a_payload = st.session_state.scenario_a_payload
    scenario_a_result = st.session_state.scenario_a_result
    scenario_b_payload = st.session_state.scenario_b_payload
    scenario_b_result = st.session_state.scenario_b_result

    if scenario_a_payload and scenario_b_payload and scenario_a_result and scenario_b_result:
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(
                """
                <div class="section-card">
                    <div class="tiny-kicker">Scenario A</div>
                    <div class="section-title">Базовый сценарий</div>
                    <div class="section-subtitle">Сохранённый набор параметров и прогноз.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(format_scenario_payload(scenario_a_payload), width="stretch")

            a1, a2 = st.columns(2)
            a1.metric("Спрос A", f"{scenario_a_result['predicted_demand']:,} шт.")
            a2.metric("Выручка A", f"{scenario_a_result['predicted_revenue']:,.0f} ₽")

        with right_col:
            st.markdown(
                """
                <div class="section-card">
                    <div class="tiny-kicker">Scenario B</div>
                    <div class="section-title">Альтернативный сценарий</div>
                    <div class="section-subtitle">Сохранённый набор параметров и прогноз.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(format_scenario_payload(scenario_b_payload), width="stretch")

            b1, b2 = st.columns(2)
            b1.metric("Спрос B", f"{scenario_b_result['predicted_demand']:,} шт.")
            b2.metric("Выручка B", f"{scenario_b_result['predicted_revenue']:,.0f} ₽")

        cmp = compare_scenarios(scenario_a_result, scenario_b_result)

        st.markdown("### Разница между сценариями")

        d1, d2 = st.columns(2)
        d1.metric(
            "Δ спроса (B vs A)",
            f"{cmp['demand_diff']:+,.0f} шт.",
            delta=f"{cmp['demand_pct']:+.2f}%",
        )
        d2.metric(
            "Δ выручки (B vs A)",
            f"{cmp['revenue_diff']:+,.0f} ₽",
            delta=f"{cmp['revenue_pct']:+.2f}%",
        )

        compare_df = pd.DataFrame(
            {
                "Сценарий": ["A", "B"],
                "Спрос": [
                    scenario_a_result["predicted_demand"],
                    scenario_b_result["predicted_demand"],
                ],
                "Выручка": [
                    scenario_a_result["predicted_revenue"],
                    scenario_b_result["predicted_revenue"],
                ],
            }
        )

        fig_compare = px.bar(
            compare_df,
            x="Сценарий",
            y=["Спрос", "Выручка"],
            barmode="group",
            title="Сравнение сценариев A и B",
        )
        style_figure(fig_compare)
        st.plotly_chart(fig_compare, width="stretch")

        if cmp["revenue_diff"] > 0 and cmp["demand_diff"] > 0:
            st.success("Сценарий B выглядит предпочтительнее: он улучшает и спрос, и выручку.")
        elif cmp["revenue_diff"] > 0:
            st.info("Сценарий B повышает выручку, но требует проверки влияния на спрос.")
        elif cmp["demand_diff"] > 0:
            st.info("Сценарий B повышает спрос, но рост выручки не подтверждён.")
        else:
            st.warning("Сценарий A выглядит сильнее по текущим расчётам.")
    else:
        st.info("Сохрани два сценария, чтобы увидеть сравнение A/B.")

with tab4:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Model Intelligence</div>
            <div class="section-title">Объяснение модели</div>
            <div class="section-subtitle">Метрики, кросс-валидация и глобальная важность признаков.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Сравнение моделей")

    try:
        meta = load_metadata()
        comparison_rows = []
        for model_name, m in meta["metrics"].items():
            cv = meta.get("cv_results", {}).get(model_name, {})
            comparison_rows.append(
                {
                    "Модель": model_name,
                    "MAE (test)": m["mae"],
                    "RMSE (test)": m["rmse"],
                    "R² (test)": m["r2"],
                    "CV MAE (avg)": cv.get("cv_avg_mae", "—"),
                    "CV RMSE (avg)": cv.get("cv_avg_rmse", "—"),
                    "CV R² (avg)": cv.get("cv_avg_r2", "—"),
                }
            )

        comp_df = pd.DataFrame(comparison_rows).sort_values("R² (test)", ascending=False)
        st.dataframe(comp_df, width="stretch")
        st.caption("Сравнение моделей по качеству на тестовой выборке и временной кросс-валидации.")

        best_model = comp_df.iloc[0]["Модель"]
        st.success(
            f"Лучшая модель по тестовой выборке: {best_model.upper()} "
            f"(R²={comp_df.iloc[0]['R² (test)']})"
        )

        fig_cmp = px.bar(
            comp_df,
            x="Модель",
            y="R² (test)",
            color="R² (test)",
            color_continuous_scale="Tealgrn",
            title="R² по моделям",
            text="R² (test)",
        )
        fig_cmp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        style_figure(fig_cmp)
        st.plotly_chart(fig_cmp, width="stretch")

    except Exception as e:
        st.warning(f"Не удалось загрузить метрики моделей: {e}")

    st.markdown("### Временная кросс-валидация")
    st.caption("Каждая модель валидируется на временных фолдах: обучение только на прошлом, проверка на будущем.")

    cv_df = load_cv_results()
    if cv_df is not None:
        fig_cv_r2 = px.line(
            cv_df,
            x="fold",
            y="r2",
            color="model",
            markers=True,
            title="R² по временным фолдам",
            labels={"fold": "Фолд", "r2": "R²", "model": "Модель"},
        )
        style_figure(fig_cv_r2)
        st.plotly_chart(fig_cv_r2, width="stretch")

        fig_cv_mae = px.line(
            cv_df,
            x="fold",
            y="mae",
            color="model",
            markers=True,
            title="MAE по временным фолдам",
            labels={"fold": "Фолд", "mae": "MAE", "model": "Модель"},
        )
        style_figure(fig_cv_mae)
        st.plotly_chart(fig_cv_mae, width="stretch")
    else:
        st.info("CV-результаты не найдены. Запусти python -m ml.train")

    st.markdown("### Глобальная важность признаков (SHAP)")
    st.caption("SHAP показывает средний вклад каждого признака в прогноз модели XGBoost.")

    try:
        with st.spinner("Вычисление SHAP-значений..."):
            shap_values, feature_columns = compute_shap_values()

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "Признак": feature_columns,
                "Важность (SHAP)": mean_shap,
            }
        ).sort_values("Важность (SHAP)", ascending=False)

        fig_shap = px.bar(
            shap_df,
            x="Важность (SHAP)",
            y="Признак",
            orientation="h",
            title="Глобальная важность признаков",
            color="Важность (SHAP)",
            color_continuous_scale="Tealgrn",
        )
        fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})
        style_figure(fig_shap)
        st.plotly_chart(fig_shap, width="stretch")

        with st.expander("Подробнее о метриках и SHAP"):
            st.write(
                "R² показывает долю объяснённой вариации, MAE — среднюю абсолютную ошибку, "
                "RMSE сильнее штрафует большие ошибки. SHAP отражает вклад признаков в итоговый прогноз."
            )

    except Exception as e:
        st.error(f"Ошибка при вычислении SHAP: {e}")
        st.info("Убедись, что установлены пакеты shap и xgboost, а модель переобучена на актуальном датасете.")

with tab5:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Tracking Layer</div>
            <div class="section-title">История прогнозов</div>
            <div class="section-subtitle">Журнал всех рассчитанных прогнозов с быстрой аналитикой и выгрузкой CSV.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_df = load_prediction_history()

    if history_df.empty:
        st.info("История прогнозов пока пуста. Сделай хотя бы один прогноз.")
    else:
        display_df = history_df.sort_values("created_at", ascending=False).copy()

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего прогнозов", len(display_df))

        if "predicted_demand" in display_df.columns:
            col2.metric("Средний спрос", f"{display_df['predicted_demand'].mean():,.1f}")

        if "predicted_revenue" in display_df.columns:
            col3.metric("Средняя выручка", f"{display_df['predicted_revenue'].mean():,.0f} ₽")

        st.dataframe(display_df, width="stretch")
        st.caption("История содержит все успешные прогнозы, сохранённые в CSV.")

        if "region" in display_df.columns and "predicted_demand" in display_df.columns:
            region_hist = (
                display_df.groupby("region")["predicted_demand"]
                .mean()
                .reset_index()
                .sort_values("predicted_demand", ascending=False)
            )

            fig_hist_region = px.bar(
                region_hist,
                x="region",
                y="predicted_demand",
                title="Средний прогноз спроса по регионам",
                labels={"region": "Регион", "predicted_demand": "Средний спрос"},
                color="predicted_demand",
                color_continuous_scale="Tealgrn",
            )
            style_figure(fig_hist_region)
            st.plotly_chart(fig_hist_region, width="stretch")

        csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                label="⬇️ Скачать историю прогнозов",
                data=csv_bytes,
                file_name="prediction_history.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_b:
            if st.button("🗑️ Очистить историю прогнозов", type="secondary", width="stretch"):
                clear_prediction_history()
                st.success("История прогнозов очищена.")
                st.rerun()

with tab6:
    st.markdown(
        """
        <div class="section-card">
            <div class="tiny-kicker">Model Registry</div>
            <div class="section-title">Информация о модели</div>
            <div class="section-subtitle">Служебные сведения о версии модели, метриках и признаках.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        meta = load_metadata()

        c1, c2 = st.columns(2)
        c1.metric("Лучшая модель", meta["best_model"].upper())
        c2.metric("Версия модели", meta["version"])

        st.markdown("#### Метрики моделей")
        metrics_rows = []
        for model_name, m in meta["metrics"].items():
            metrics_rows.append(
                {
                    "Модель": model_name,
                    "MAE": m["mae"],
                    "RMSE": m["rmse"],
                    "R²": m["r2"],
                    "CV MAE": meta.get("cv_results", {}).get(model_name, {}).get("cv_avg_mae", "—"),
                    "CV RMSE": meta.get("cv_results", {}).get(model_name, {}).get("cv_avg_rmse", "—"),
                    "CV R²": meta.get("cv_results", {}).get(model_name, {}).get("cv_avg_r2", "—"),
                }
            )

        st.dataframe(pd.DataFrame(metrics_rows), width="stretch")
        st.caption("Сводная таблица качества всех обученных моделей.")

        st.markdown("#### Используемые признаки")
        st.code("\n".join(meta["feature_columns"]))

        colx, coly = st.columns(2)
        with colx:
            st.markdown("#### Доступные регионы")
            st.write(", ".join(meta["regions"]))
        with coly:
            st.markdown("#### Типы магазинов")
            st.write(", ".join(meta["store_types"]))

        if "dataset" in meta:
            st.markdown("#### Параметры датасета")
            dataset_info = pd.DataFrame([meta["dataset"]])
            st.dataframe(dataset_info, width="stretch")

    except FileNotFoundError:
        st.warning("Не найден metadata.json. Сначала выполни `python -m ml.train`.")