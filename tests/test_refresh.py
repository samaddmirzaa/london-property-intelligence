import pandas as pd
import pytest

from london_property import refresh
from london_property.tools import area_dir, london


@pytest.fixture
def profile():
    return pd.read_parquet(area_dir/'ward_profile.parquet').set_index('ward_code')


@pytest.fixture
def offline(monkeypatch):
    """No downloads in tests: every source fails unless a test swaps in its own data."""
    def fail():
        raise OSError('no connection')
    for name in ['latest_crime', 'latest_hpi', 'latest_rents']:
        monkeypatch.setattr(refresh, name, fail)


def fake_hpi(date):
    return pd.DataFrame({'Date': [date], 'AreaCode': [london], 'AveragePrice': [600000], '12m%Change': [2.0],
                         'FlatPrice': [450000], 'TerracedPrice': [650000], 'SemiDetachedPrice': [800000],
                         'DetachedPrice': [1200000]})


def test_crime_rates():
    crime = pd.DataFrame({'WardCode': ['W1', 'W1', 'W2', 'W2'], 'Group': ['THEFT', 'NFIB FRAUD', 'THEFT', 'BURGLARY'],
                          '202501': [10, 50, 4, 1], '202502': [10, 50, 6, 1]})
    rates, months = refresh.crime_rates(crime, pd.Series({'W1': 1000, 'W2': 2000, london: 3000}))
    assert months == '202501-202502'
    assert rates.loc['W1', 'crime_total'] == 20    # 20 thefts per 1,000 residents; the fraud is left out
    assert rates.loc['W2', 'crime_theft'] == 5
    assert rates.loc[london, 'crime_total'] == pytest.approx(32 / 3000 * 1000, abs=0.01)


def test_refresh_updates_newer_data_and_keeps_the_rest(profile, offline, monkeypatch):
    before = profile.copy()
    monkeypatch.setattr(refresh, 'latest_hpi', lambda: fake_hpi('01/12/2099'))
    report = refresh.refresh(profile)
    assert report[1] == 'prices: updated to 2099-12'
    assert profile.loc[london, 'price_avg'] == 600000
    assert 'kept the current data' in report[0] and 'kept the current data' in report[2]
    unchanged = [c for c in profile.columns if c.startswith(('crime', 'rent'))]
    pd.testing.assert_frame_equal(profile[unchanged], before[unchanged])


def test_refresh_ignores_older_data(profile, offline, monkeypatch):
    before = profile.copy()
    monkeypatch.setattr(refresh, 'latest_hpi', lambda: fake_hpi('01/01/2020'))
    report = refresh.refresh(profile)
    assert report[1] == 'prices: nothing newer than 2020-01'
    pd.testing.assert_frame_equal(profile, before)