import pytest
import pandas as pd

from london_property.tools import (
    load_area_data, normalise_postcode, get_area_profile, get_neighbourhood, find_nearby, get_guide, council_tax,
    stamp_duty, london_percentile, place_types, walk_m
)


@pytest.fixture(scope='module')
def data():
    return load_area_data()


def test_normalise_postcode():
    assert normalise_postcode('e83rl') == 'E8 3RL'
    assert normalise_postcode(' sw1a 1aa ') == 'SW1A 1AA'


def test_area_profile(data):
    result = get_area_profile(data, 'sw1a1aa')
    assert result['postcode'] == 'SW1A 1AA'
    assert result['location']['borough_name'] == 'Westminster'
    assert result['ward']['ward_name'] == result['location']['ward_name']
    assert result['london_average']['population'] > 8_000_000


def test_unknown_postcode(data):
    with pytest.raises(ValueError):
        get_area_profile(data, 'ZZ69 69ZZ')

def test_find_nearby(data):
    result = find_nearby(data, 'N16 0UD', 'park', radius_m=1000, limit=100)
    distances = [p['distance_m'] for p in result['nearest']]
    assert result['count'] == len(result['nearest']) > 0
    assert distances == sorted(distances)
    assert max(distances) <= 1000
    assert [p['name'] for p in result['nearest']].count('Clissold Park') == 1


def test_find_nearby_subtype_and_bad_type(data):
    result = find_nearby(data, 'N16 0UD', 'school', subtype='primary')
    assert all(p['subtype'] == 'primary' for p in result['nearest'])
    with pytest.raises(ValueError):
        find_nearby(data, 'N16 0UD', 'castle')

def test_neighbourhood(data):
    result = get_neighbourhood(data, 'n16 0ud')
    walk = result['within_10_min_walk']
    assert set(walk) == set(place_types)
    assert walk['cafe']['count'] == find_nearby(data, 'N16 0UD', 'cafe', radius_m=walk_m)['count']
    assert all(0 <= v['london_percentile'] <= 100 for v in walk.values())
    assert 0 <= result['ward_vs_london']['crime_total']['london_percentile'] <= 100
    assert 0 <= result['here_vs_london']['no2']['london_percentile'] <= 100
    assert all(pct >= 80 for pct in result['higher_than_most_of_london'].values())
    assert all(pct <= 20 for pct in result['lower_than_most_of_london'].values())
    nearest = find_nearby(data, 'N16 0UD', 'school', radius_m=3000, limit=1, subtype='primary')['nearest'][0]
    assert result['nearest']['primary school']['name'] == nearest['name']

def test_neighbourhood_city_of_london(data):
    result = get_neighbourhood(data, 'EC2V 7HH')
    assert result['ward_vs_london']['crime_total']['ward'] is None
    assert result['ward_vs_london']['crime_total']['london_percentile'] is None

@pytest.mark.parametrize('guide, question, expected', [
    ('leasehold-property', 'ground rent', 'Service charges and other expenses'),
    ('leasehold-property', 'extend my lease', 'Extending, changing or ending a lease'),
    ('buying-a-home', 'conveyancing solicitor', 'Transferring ownership (conveyancing)'),
    ('buying-a-home', 'energy performance certificate', 'Deciding on a home'),
    ('shared-ownership-scheme', 'staircasing buying more shares', "Buying more shares ('staircasing')"),
    ('right-to-buy-buying-your-council-home', 'discount', 'Discounts'),
    ('council-tax', 'challenge my band', 'Working out your Council Tax'),
    ('joint-property-ownership', 'buying with my partner', 'Overview'),
])
def test_get_guide_finds_the_right_section(data, guide, question, expected):
    result = get_guide(data, guide, question)
    assert expected in [s['part_title'] for s in result['sections']]

def test_get_guide_result(data):
    result = get_guide(data, 'lifetime-isa', 'zzz')
    assert result['sections'][0]['part_title'] == 'Overview'
    assert len(result['sections']) + len(result['other_sections']) == 3
    with pytest.raises(ValueError):
        get_guide(data, 'how-to-get-rich', 'property')

def test_council_tax(data):
    band_d = council_tax(data, 'N16 0UD', 'D')['annual_bill']
    assert council_tax(data, 'N16 0UD', 'h')['annual_bill'] == round(band_d * 2, 2)
    with pytest.raises(ValueError):
        council_tax(data, 'N16 0UD', 'Z')


@pytest.mark.parametrize('price, ftb, additional, expected', [
    (500_000, False, False, 15_000),
    (500_000, True, False, 10_000),
    (300_000, True, False, 0),
    (600_000, True, False, 20_000),
    (500_000, False, True, 40_000),
    (1_000_000, False, False, 43_750),
])
def test_stamp_duty(price, ftb, additional, expected):
    assert stamp_duty(price, first_time_buyer=ftb, additional_property=additional)['stamp_duty'] == expected

def test_london_percentile():
    table = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [0, 0, 0, 0, 5], 'c': [1, None, 2, 3, None]})
    row = pd.Series({'a': 5.0, 'b': 0.0, 'c': 2.0})
    assert london_percentile(table, row).tolist() == [90, 40, 50]

def test_stamp_duty_bands():
    result = stamp_duty(650_000, additional_property=True, non_uk_resident=True)
    assert [b['rate_pct'] for b in result['bands']] == [7, 9, 12]
    assert sum(b['tax'] for b in result['bands']) == result['stamp_duty'] == 68_000