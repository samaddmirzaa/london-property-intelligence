from fastapi.testclient import TestClient
from london_property.api import app

client = TestClient(app)


def test_area_profile():
    response = client.get('/area/sw1a1aa')
    assert response.status_code == 200
    assert response.json()['location']['borough_name'] == 'Westminster'


def test_area_unknown_postcode():
    response = client.get('/area/ZZ69 69ZZ')
    assert response.status_code == 404
    assert 'not a live London postcode' in response.json()['detail']


def test_nearby():
    response = client.get('/nearby/N16 0UD', params={'type': 'school', 'subtype': 'primary', 'limit': 3})
    assert response.status_code == 200
    body = response.json()
    assert body['type'] == 'school'
    assert 0 < len(body['nearest']) <= 3


def test_nearby_rejects_bad_input():
    assert client.get('/nearby/N16 0UD', params={'type': 'castle'}).status_code == 422
    assert client.get('/nearby/N16 0UD', params={'type': 'school', 'radius_m': 10000}).status_code == 422
    assert client.get('/nearby/N16 0UD').status_code == 422


def test_council_tax():
    response = client.get('/council-tax/N16 0UD', params={'band': 'D'})
    assert response.status_code == 200
    assert response.json()['annual_bill'] == response.json()['band_d_bill']
    assert client.get('/council-tax/N16 0UD', params={'band': 'Z'}).status_code == 422


def test_stamp_duty():
    response = client.get('/stamp-duty', params={'price': 500000, 'first_time_buyer': True})
    assert response.status_code == 200
    assert response.json()['stamp_duty'] == 10000
    assert client.get('/stamp-duty', params={'price': 500000, 'first_time_buyer': True,
                                             'additional_property': True}).status_code == 422