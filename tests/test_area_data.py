from pathlib import Path

import pandas as pd
import pytest

area_dir = Path(__file__).resolve().parents[1]/'data'/'processed'/'area'
london = 'E12000007'


@pytest.fixture(scope='module')
def postcodes():
    return pd.read_parquet(area_dir/'postcodes.parquet')


@pytest.fixture(scope='module')
def profile():
    return pd.read_parquet(area_dir/'ward_profile.parquet').set_index('ward_code')


@pytest.fixture(scope='module')
def points():
    return pd.read_parquet(area_dir/'points.parquet')

@pytest.fixture(scope='module')
def walk_benchmark():
    return pd.read_parquet(area_dir/'walk_benchmark.parquet')

def test_postcodes_cover_london(postcodes):
    assert len(postcodes) > 170_000
    assert postcodes['postcode'].is_unique
    assert postcodes['borough_code'].nunique() == 33
    assert postcodes['ward_code'].nunique() == 704


def test_postcode_values_valid(postcodes):
    assert postcodes['flood_zone'].isin([1, 2, 3]).all()
    assert postcodes['ptal'].isin(['0', '1a', '1b', '2', '3', '4', '5', '6a', '6b']).all()
    assert postcodes[['no2', 'pm25', 'pm10']].notna().all().all()
    assert postcodes['imd_decile'].dropna().between(1, 10).all()


def test_every_postcode_ward_has_a_profile(postcodes, profile):
    assert postcodes['ward_code'].isin(profile.index).all()


def test_profile_shape(profile):
    assert len(profile) == 705
    assert profile.index.is_unique
    assert profile.loc[london, 'population'] > 8_000_000


def test_profile_values_valid(profile):
    crime = profile.filter(like='crime_').drop(columns='crime_months')
    assert (crime.dropna() >= 0).all().all()
    ages = profile.filter(like='pct_age_').dropna().sum(axis=1)
    assert ages.between(99.5, 100.5).all()
    assert profile['council_tax_band_d'].between(500, 5000).all()


def test_points_valid(points):
    assert points[['type', 'subtype', 'id', 'name', 'info']].notna().all().all()
    assert points['lat'].between(51.28, 51.70).all()
    assert points['lon'].between(-0.52, 0.34).all()
    expected = {'school', 'gp', 'supermarket', 'park', 'listed building', 'restaurant', 'cafe',
                'pub or bar', 'gym', 'dentist', 'takeaway', 'bank', 'pharmacy', 'childcare',
                'library', 'cinema', 'theatre'}
    assert set(points['type']) == expected

def test_walk_benchmark_valid(walk_benchmark, points):
    assert len(walk_benchmark) == 5000
    assert set(walk_benchmark.columns) == set(points['type'])
    assert (walk_benchmark >= 0).all().all()

def test_school_dates_unambiguous(points):
    assert not points['info'].str.contains(r'\d{2}/\d{2}/\d{4}').any()