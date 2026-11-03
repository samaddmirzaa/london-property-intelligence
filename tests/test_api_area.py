from fastapi.testclient import TestClient
from london_property.api import app

client = TestClient(app)


def test_neighbourhood():
    response = client.get('/neighbourhood/N16 0UD')
    assert response.status_code == 200
    assert response.json()['within_10_min_walk']['cafe']['count'] > 0
    assert client.get('/neighbourhood/ZZ69 69ZZ').status_code == 404

def test_guide():
    response = client.get('/guide/leasehold-property', params={'question': 'ground rent'})
    assert response.status_code == 200
    assert response.json()['sections'][0]['url'].startswith('https://www.gov.uk/leasehold-property/')
    assert client.get('/guide/not-a-guide', params={'question': 'rent'}).status_code == 422

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