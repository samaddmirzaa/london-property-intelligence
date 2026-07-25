from fastapi.testclient import TestClient
from london_property.api import app

client = TestClient(app)

Valid = {
    'postcode': 'SW1A 1AA',
    'floor_area': 70,
    'property_type': 'Flat',
    'tenure': 'Freehold',
    'built_year': 1960
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {'Status': 'OK'}

def test_predict_valid_request():
    response = client.post("/predict-price", json=Valid)
    assert response.status_code == 200
    body = response.json()
    assert 50000 < body['predicted_price'] < 10000000
    assert body['district'] == 'SW1A'

def test_predict_unknown_postcode():
    payload = Valid | {'postcode': 'ZZ69 69ZZ'}
    response = client.post("/predict_price", json=payload)
    assert response.status_code == 404

def test_predict_rejects_invalid_input():
    assert client.post('/predict-price', json=Valid | {'floor_area': 5}).status_code == 422
    assert client.post('/predict-price', json=Valid | {'property_type': 'Castle'}).status_code == 422

