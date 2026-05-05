from types import SimpleNamespace
from app.api.services.model_service import ModelService


def test_model_service_predict_returns_dict():
    service = ModelService()

    payload = SimpleNamespace(
        store_id=10,
        region="Москва",
        store_type="Супермаркет",
        month=7,
        weekday=5,
        day_of_year=200,
        is_weekend=True,
        is_holiday=False,
        promo_flag=False,
        temperature=28.5,
        price=58.0,
    )

    result = service.predict(payload)

    assert isinstance(result, dict)
    assert "predicted_demand" in result
    assert "predicted_revenue" in result
    assert "model_name" in result
    assert "model_version" in result


def test_model_service_predict_values_are_positive():
    service = ModelService()

    payload = SimpleNamespace(
        store_id=25,
        region="Москва",
        store_type="Супермаркет",
        month=7,
        weekday=5,
        day_of_year=200,
        is_weekend=True,
        is_holiday=False,
        promo_flag=True,
        temperature=30.0,
        price=60.0,
    )

    result = service.predict(payload)

    assert result["predicted_demand"] >= 0
    assert result["predicted_revenue"] >= 0


def test_model_service_invalid_region_raises_error():
    service = ModelService()

    payload = SimpleNamespace(
        store_id=25,
        region="Луна",
        store_type="Супермаркет",
        month=7,
        weekday=5,
        day_of_year=200,
        is_weekend=True,
        is_holiday=False,
        promo_flag=False,
        temperature=25.0,
        price=58.0,
    )

    try:
        service.predict(payload)
        assert False, "Ожидалась ошибка ValueError"
    except ValueError:
        assert True