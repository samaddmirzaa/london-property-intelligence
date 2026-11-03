"""Monthly refresh of the data that changes every month: crime, house prices, rents and the GOV.UK guides.

Downloads the latest release of each source, recomputes those columns of the ward profile with the same
functions the full build uses, and keeps the last good data for any source that fails or has nothing new.
Run by the monthly GitHub Action
"""
import io
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from london_property import guide_data
from london_property.tools import area_dir, london

crime_page = 'https://data.london.gov.uk/dataset/mps-recorded-crime-geographic-breakdown-exy3m'
crime_link = re.compile(r'download/[\w-]+/\w+/MPS%20Ward%20Level%20Crime%20\(most%20recent%2024%20months\)\.csv')
hpi_url = 'https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-{}.csv'
rents_page = 'https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics'
rents_link = re.compile(r'/file\?uri=[^"]+?/(\d{1,2}[a-z]+\d{4})/[^"]+?\.xlsx')


def crime_rates(crime, population):
    """Crimes per 1,000 residents over the last 12 months for each Met ward and London (fraud excluded),
    and the months covered, e.g. '202510-202609'."""
    crime = crime[crime['WardCode'].isin(population.index) & ~crime['Group'].isin(['NFIB FRAUD', 'FRAUD AND FORGERY'])]
    months = sorted(c for c in crime.columns if c.isdigit())[-12:]
    crime = crime.assign(crimes_12m=crime[months].sum(axis=1))
    counts = crime.pivot_table(index='WardCode', columns='Group', values='crimes_12m', aggfunc='sum', fill_value=0)
    counts.columns = 'crime_' + counts.columns.str.lower().str.replace(' ', '_')
    counts['crime_total'] = counts.sum(axis=1)
    met_pop = population.loc[counts.index]
    rates = counts.div(met_pop, axis=0) * 1000
    rates.loc[london] = counts.sum() / met_pop.sum() * 1000
    return rates.round(2), f'{months[0]}-{months[-1]}'


def borough_prices(hpi):
    """Average prices by property type from the latest month of the UK House Price Index, by area code."""
    hpi = hpi.assign(Date=pd.to_datetime(hpi['Date'], dayfirst=True))
    latest = hpi[hpi['Date'] == hpi['Date'].max()]
    prices = latest.set_index('AreaCode')[['AveragePrice', '12m%Change', 'FlatPrice', 'TerracedPrice',
                                            'SemiDetachedPrice', 'DetachedPrice']]
    prices.columns = ['price_avg', 'price_change_12m', 'price_flat', 'price_terraced', 'price_semi', 'price_detached']
    prices['price_month'] = latest['Date'].iloc[0].strftime('%Y-%m')
    return prices


def borough_rents(rents):
    """Average rents by bedrooms from the latest month of the ONS Price Index of Private Rents, by area code.
    English areas only: the workbook's Northern Ireland rows have no area codes."""
    latest = rents[(rents['Time period'] == rents['Time period'].max()) & rents['Area code'].str.startswith('E')]
    rent = latest.set_index('Area code')[['Rental price', 'Annual change', 'Rental price one bed',
                                          'Rental price two bed', 'Rental price three bed',
                                          'Rental price four or more bed']]
    rent = rent.apply(pd.to_numeric, errors='coerce')    # ONS marks missing values as [x]
    rent.columns = ['rent_avg', 'rent_change_12m', 'rent_1bed', 'rent_2bed', 'rent_3bed', 'rent_4bed_plus']
    rent['rent_change_12m'] = rent['rent_change_12m'].round(1)
    rent['rent_month'] = latest['Time period'].iloc[0][:7]
    return rent


def download(url):
    return urlopen(Request(url, headers={'User-Agent': 'london-property-intelligence monthly refresh'}), timeout=300)


def latest_crime():
    """The Met's ward-level crime file for the latest 24 months. The page also lists older releases under the
    same name, so every candidate is read and the one reaching the latest month wins."""
    page = download(crime_page).read().decode()
    files = [pd.read_csv(download(f'https://data.london.gov.uk/{link}')) for link in sorted(set(crime_link.findall(page)))]
    return max(files, key=lambda f: max(c for c in f.columns if c.isdigit()))


def latest_hpi():
    """The newest UK House Price Index full file: the name includes the month, so try recent months in turn."""
    month = pd.Timestamp.today().to_period('M')
    for _ in range(6):
        try:
            return pd.read_csv(download(hpi_url.format(month)))
        except URLError:
            month -= 1
    raise URLError('no UK HPI full file in the last 6 months')


def latest_rents():
    """The newest ONS private rents workbook. Each release sits in a folder named by its date (21october2026)."""
    page = download(rents_page).read().decode()
    link = max(rents_link.finditer(page), key=lambda m: pd.to_datetime(m[1], format='%d%B%Y'))[0]
    rents = pd.read_excel(io.BytesIO(download(f'https://www.ons.gov.uk{link}').read()), sheet_name='Table 1', skiprows=2)
    return rents.assign(**{'Time period': pd.to_datetime(rents['Time period']).dt.strftime('%Y-%m-%d')})


def refresh(profile):
    """Update the monthly columns of the ward profile in place. Returns a line per source saying what happened."""
    report = []
    try:
        rates, months = crime_rates(latest_crime(), profile['population'])
        if months[-6:] > profile['crime_months'].max()[-6:]:
            profile[rates.columns] = rates
            profile['crime_months'] = months
            report.append(f'crime: updated to {months}')
        else:
            report.append(f'crime: nothing newer than {months}')
    except Exception as e:
        report.append(f'crime: kept the current data ({e!r})')

    for name, load, compute in [('price', latest_hpi, borough_prices), ('rent', latest_rents, borough_rents)]:
        try:
            borough = compute(load())
            month = borough[f'{name}_month'].iloc[0]
            if month > profile[f'{name}_month'].max():    # max, as City of London has no rents
                for col in borough.columns:
                    profile[col] = profile['borough_code'].map(borough[col])
                report.append(f'{name}s: updated to {month}')
            else:
                report.append(f'{name}s: nothing newer than {month}')
        except Exception as e:
            report.append(f'{name}s: kept the current data ({e!r})')
    return report


def main():
    path = area_dir/'ward_profile.parquet'
    profile = pd.read_parquet(path).set_index('ward_code')
    before = profile.copy()
    report = refresh(profile)
    if not profile.equals(before):    # rewrite the file only when the data changed
        profile.reset_index().to_parquet(path, index=False)
    try:
        guide_data.main()
        report.append('guides: downloaded again')
    except Exception as e:
        report.append(f'guides: kept the current guides ({e!r})')
    print('\n'.join(report))


if __name__ == '__main__':
    main()