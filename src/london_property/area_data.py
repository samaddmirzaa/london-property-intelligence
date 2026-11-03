"""Rebuild the three area tables the chatbot uses from the raw files in data/raw.

Run from the project root:  python -m london_property.area_data
Outputs (data/processed/area): postcodes.parquet, ward_profile.parquet, points.parquet
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from london_property.tools import count_by_type, walk_m

project_root = Path(__file__).resolve().parents[2]
raw = project_root/'data'/'raw'
postcode_file = project_root/'data'/'supplementary'/'london_postcodes.csv'
out_dir = project_root/'data'/'processed'/'area'

bng = 'EPSG:27700'
london = 'E12000007'

# Overture's coarse category is our type (every cuisine is 'restaurant', every pub or bar is 'bar')
place_types = {
    'restaurant': 'restaurant', 'fast_food_restaurant': 'takeaway',
    'cafe': 'cafe', 'coffee_shop': 'cafe', 'bar': 'pub or bar',
    'gym': 'gym', 'fitness_studio': 'gym',
    'pharmacy_and_drug_store': 'pharmacy', 'dental_clinic': 'dentist',
    'preschool': 'childcare', 'library': 'library',
    'movie_theater': 'cinema', 'theatre_venue': 'theatre', 'bank_or_credit_union': 'bank',
}

# Ofsted: pre-September-2024 grade codes, and the six report card areas (since November 2025)
grade = {'1': 'Outstanding', '2': 'Good', '3': 'Requires improvement', '4': 'Inadequate'}
report_cols = ['Inclusion', 'Curriculum and teaching', 'Achievement', 'Attendance and behaviour',
               'Personal development and wellbeing', 'Leadership and governance']


def load_zones(name):
    """Planning shapes still in force, repaired and in British National Grid."""
    zones = gpd.read_file(raw/'planning'/f'{name}.geojson').to_crs(bng)
    zones = zones[zones['end-date'].isna() | (zones['end-date'] == '')]
    return zones.set_geometry(zones.geometry.make_valid())


def to_ward(df, lsoa_col, value_cols, lookup):
    """Sum LSOA counts up to wards, plus a London total row."""
    df = df.merge(lookup[['LSOA21CD', 'WD24CD']], left_on=lsoa_col, right_on='LSOA21CD')
    ward = df.groupby('WD24CD')[value_cols].sum()
    ward.loc[london] = ward.sum()
    return ward


def school_level(low, high):
    """Nursery / primary / secondary / all-through / sixth form, from the ages a school takes."""
    if high <= 5:
        return 'nursery'
    if low >= 16:
        return 'sixth form'
    if high <= 13:
        return 'primary'
    if low >= 9:
        return 'secondary'
    return 'all-through'


def uk_date(text):
    """'11/10/2023' (Ofsted's day-first format) -> '11 Oct 2023', which no reader can misread as 10 November."""
    d = pd.to_datetime(text, format='%d/%m/%Y')
    return f'{d.day} {d:%b %Y}'


def ofsted_summary(r):
    """Every Ofsted inspection type a school has, each with its date."""
    found = []
    if pd.notna(r['Inclusion']):
        grades = ', '.join(f'{c}: {r[c]}' for c in report_cols)
        found.append(f"Report card ({uk_date(r['Inspection start date'])}): {grades}, "
                     f"Safeguarding: {r['Safeguarding standards']}")
    overall = r['Latest OEIF overall effectiveness']
    if pd.notna(overall):
        quality = str(r['Latest OEIF quality of education']).split('.')[0]
        found.append(f"Graded inspection ({uk_date(r['Inspection start date of latest OEIF graded inspection'])}): "
                     f"{grade.get(overall, overall)} overall, quality of education {grade.get(quality, quality)}")
    if pd.notna(r['Ungraded inspection overall outcome']):
        found.append(f"Ungraded inspection ({uk_date(r['Date of latest ungraded inspection'])}): "
                     f"{r['Ungraded inspection overall outcome']}")
    return ' | '.join(found) or 'Not yet inspected'

def school_info(r):
    """One readable sentence about a school for the chatbot."""
    info = [r['level'].capitalize(), r['TypeOfEstablishment (name)'],
            f"ages {int(r['StatutoryLowAge'])}-{int(r['StatutoryHighAge'])}" if pd.notna(r['StatutoryLowAge']) else None,
            f"{int(r['NumberOfPupils'])} pupils" if pd.notna(r['NumberOfPupils']) else None,
            r['Gender (name)'] if r['Gender (name)'] in ['Girls', 'Boys'] else None,
            r['ReligiousCharacter (name)'] if r['ReligiousCharacter (name)'] not in ['Does not apply', 'None'] else None]
    return ', '.join(i for i in info if isinstance(i, str)) + '. Ofsted: ' + r['ofsted']


def build_postcodes():
    """One row per live London postcode with everything tied to its exact location."""
    cols = {
        'Postcode': 'postcode', 'In Use?': 'in_use',
        'Latitude': 'lat', 'Longitude': 'lon', 'Easting': 'easting', 'Northing': 'northing',
        'LSOA21 Code': 'lsoa_code', 'Ward Code': 'ward_code', 'Ward': 'ward_name',
        'District Code': 'borough_code', 'District': 'borough_name',
        'Nearest station': 'nearest_station', 'Distance to station': 'station_km',
    }
    pc = pd.read_csv(postcode_file, usecols=list(cols)).rename(columns=cols)
    pc = pc[pc['in_use'] == 'Yes'].drop(columns='in_use').dropna().reset_index(drop=True)
    pc = pc.astype({'easting': int, 'northing': int})

    points = gpd.GeoDataFrame(pc[['postcode']], geometry=gpd.points_from_xy(pc['easting'], pc['northing']), crs=bng)

    hits = gpd.sjoin(points, load_zones('planning_flood_risk_zones')[['flood-risk-level', 'geometry']], predicate='within')
    pc['flood_zone'] = pc['postcode'].map(hits.groupby('postcode')['flood-risk-level'].max()).fillna('1').astype(int)

    hits = gpd.sjoin(points, load_zones('planning_conservation_areas')[['name', 'geometry']], predicate='within')
    pc['conservation_area'] = pc['postcode'].map(hits.groupby('postcode')['name'].first())

    hits = gpd.sjoin(points, load_zones('planning_article4_areas')[['name', 'geometry']], predicate='within')
    pc['article4'] = pc['postcode'].map(hits.groupby('postcode')['name'].agg(lambda s: '; '.join(sorted(set(s)))))

    ptal = pd.read_csv(raw/'transport'/'transport_ptal_100m_grid_2023.csv')
    grid = gpd.points_from_xy(ptal['lon'], ptal['lat'], crs='EPSG:4326').to_crs(bng)
    _, idx = cKDTree(list(zip(grid.x, grid.y))).query(pc[['easting', 'northing']].to_numpy())
    pc['ptal'] = ptal['PTAL_2023'].to_numpy()[idx]
    pc['ptal_ai'] = ptal['AI'].to_numpy()[idx].round(1)

    air = pd.read_csv(raw/'air'/'air_quality_no2_pm25_pm10_100m_grid_2022.csv')
    air = air.rename(columns={'easting': 'cell_e', 'northing': 'cell_n', 'pm10m': 'pm10'})
    pc['cell_e'] = pc['easting'] // 100 * 100 + 50
    pc['cell_n'] = pc['northing'] // 100 * 100 + 50
    pc = pc.merge(air, on=['cell_e', 'cell_n'], how='left').drop(columns=['cell_e', 'cell_n'])

    iod = pd.read_csv(raw/'deprivation'/'deprivation_iod2025_lsoa.csv')
    decile = iod.set_index('LSOA code (2021)')['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)']
    pc['imd_decile'] = pc['lsoa_code'].map(decile).astype('Int64')
    return pc


def build_ward_profile(pc, wards, lookup):
    """One row per ward plus a London row: people, crime, homes, transport, air, prices, rents, council tax."""
    profile = wards[['WD24CD', 'WD24NM', 'LAD24CD', 'LAD24NM']].copy()
    profile.columns = ['ward_code', 'ward_name', 'borough_code', 'borough_name']
    profile.loc[len(profile)] = [london, 'London', london, 'London']
    profile = profile.set_index('ward_code')

    # Population (mid-2022), the denominator for crime rates
    iod = pd.read_csv(raw/'deprivation'/'deprivation_iod2025_lsoa.csv')
    profile['population'] = to_ward(iod, 'LSOA code (2021)', ['Total population: mid 2022'], lookup)['Total population: mid 2022']

    # Crime: last 12 months per 1,000 residents, fraud excluded, Met wards only
    crime = pd.read_csv(raw/'crime'/'crime_ward_monthly_2024_10_to_2026_09.csv')
    crime = crime[crime['WardCode'].isin(profile.index) & ~crime['Group'].isin(['NFIB FRAUD', 'FRAUD AND FORGERY'])]
    months = sorted(c for c in crime.columns if c.isdigit())[-12:]
    crime = crime.assign(crimes_12m=crime[months].sum(axis=1))
    counts = crime.pivot_table(index='WardCode', columns='Group', values='crimes_12m', aggfunc='sum', fill_value=0)
    counts.columns = 'crime_' + counts.columns.str.lower().str.replace(' ', '_')
    counts['crime_total'] = counts.sum(axis=1)
    met_pop = profile.loc[counts.index, 'population']
    rates = counts.div(met_pop, axis=0) * 1000
    rates.loc[london] = counts.sum() / met_pop.sum() * 1000
    profile = profile.join(rates.round(2))
    profile['crime_months'] = f'{months[0]}-{months[-1]}'

    # Age mix (Census 2021)
    age = pd.read_csv(raw/'census'/'census2021_age_bands_lsoa.csv')
    age_cols = [c for c in age.columns if c.startswith('Age: Aged')]
    age_ward = to_ward(age, 'geography code', ['Age: Total'] + age_cols, lookup)
    bands = {'pct_age_0_14': age_cols[0:3], 'pct_age_15_29': age_cols[3:6], 'pct_age_30_44': age_cols[6:9],
             'pct_age_45_64': age_cols[9:13], 'pct_age_65_plus': age_cols[13:]}
    for name, band_cols in bands.items():
        profile[name] = (age_ward[band_cols].sum(axis=1) / age_ward['Age: Total'] * 100).round(1)

    # Housing tenure (Census 2021)
    tenure = pd.read_csv(raw/'census'/'census2021_housing_tenure_lsoa.csv')
    prefix = 'Tenure of household: '
    total = prefix + 'Total: All households'
    tenure_cols = {'Owned': 'pct_owned', 'Social rented': 'pct_social_rented', 'Private rented': 'pct_private_rented'}
    tenure_ward = to_ward(tenure, 'geography code', [total] + [prefix + c for c in tenure_cols], lookup)
    profile['households'] = tenure_ward[total]
    for col, name in tenure_cols.items():
        profile[name] = (tenure_ward[prefix + col] / tenure_ward[total] * 100).round(1)

    # Transport and air quality, averaged over each ward's postcodes
    loc_cols = ['ptal_ai', 'no2', 'pm25', 'pm10']
    local = pc.groupby('ward_code')[loc_cols].mean()
    local.loc[london] = pc[loc_cols].mean()
    profile = profile.join(local.round({'ptal_ai': 1, 'no2': 1, 'pm25': 1, 'pm10': 1}))

    # Borough house prices: latest month of the UK House Price Index
    hpi = pd.read_csv(raw/'house_prices'/'house_price_index_borough_monthly_to_2026_07.csv')
    hpi['Date'] = pd.to_datetime(hpi['Date'], dayfirst=True)
    latest = hpi[hpi['Date'] == hpi['Date'].max()]
    prices = latest.set_index('AreaCode')[['AveragePrice', '12m%Change', 'FlatPrice', 'TerracedPrice',
                                            'SemiDetachedPrice', 'DetachedPrice']]
    prices.columns = ['price_avg', 'price_change_12m', 'price_flat', 'price_terraced', 'price_semi', 'price_detached']
    prices['price_month'] = latest['Date'].iloc[0].strftime('%Y-%m')

    # Borough rents: latest month of the ONS Price Index of Private Rents
    rents = pd.read_csv(raw/'rents'/'private_rents_borough_monthly_2015_01_to_2026_08.csv')
    latest_rent = rents[rents['Time period'] == rents['Time period'].max()]
    rent = latest_rent.set_index('Area code')[['Rental price', 'Annual change', 'Rental price one bed',
                                               'Rental price two bed', 'Rental price three bed',
                                               'Rental price four or more bed']]
    rent = rent.apply(pd.to_numeric, errors='coerce')
    rent.columns = ['rent_avg', 'rent_change_12m', 'rent_1bed', 'rent_2bed', 'rent_3bed', 'rent_4bed_plus']
    rent['rent_change_12m'] = rent['rent_change_12m'].round(1)
    rent['rent_month'] = latest_rent['Time period'].iloc[0][:7]

    # Council tax Band D 2026-27; London = tax-base-weighted average
    ct = pd.read_csv(raw/'council_tax'/'council_tax_borough_2026_27.csv')
    band_d = [c for c in ct.columns if c.startswith('17.')][0]
    tax_base = '7. Council tax base for council tax setting purposes (current year)'
    tax = ct.set_index('ONS Code')[[band_d]].rename(columns={band_d: 'council_tax_band_d'})
    tax.loc[london] = (ct[band_d] * ct[tax_base]).sum() / ct[tax_base].sum()

    borough = prices.join(rent).join(tax.round(2))
    return profile.join(borough, on='borough_code')


def build_points(pc, wards):
    """One row per place a buyer might want nearby, all in the same seven columns plus lat/lon."""
    parts = []

    # Schools: open schools with an age-based level and an Ofsted summary
    gias = pd.read_csv(raw/'schools'/'schools_register_gias_2026_10_14.csv', low_memory=False)
    keep_groups = ['Local authority maintained schools', 'Academies', 'Free Schools', 'Special schools', 'Independent schools']
    schools = gias[(gias['EstablishmentStatus (name)'] == 'Open') & gias['EstablishmentTypeGroup (name)'].isin(keep_groups)]
    schools = schools.dropna(subset=['Easting', 'Northing'])
    schools = schools[schools['Easting'] > 0]
    schools = schools.assign(level=[school_level(lo, hi) for lo, hi in zip(schools['StatutoryLowAge'], schools['StatutoryHighAge'])])
    ofsted = pd.read_csv(raw/'schools'/'schools_ofsted_latest_inspections_2026_08_31.csv', dtype=str)
    ofsted['ofsted'] = ofsted.apply(ofsted_summary, axis=1)
    ofsted['URN'] = ofsted['URN'].astype(int)
    schools = schools.merge(ofsted[['URN', 'ofsted']], on='URN', how='left')
    schools['ofsted'] = schools['ofsted'].fillna('No state-school Ofsted record')
    parts.append(pd.DataFrame({
        'type': 'school', 'subtype': schools['level'], 'id': schools['URN'].astype(str),
        'name': schools['EstablishmentName'], 'info': schools.apply(school_info, axis=1),
        'easting': schools['Easting'], 'northing': schools['Northing'],
    }))

    # GP surgeries: open RO76 practices, located by postcode
    gp = pd.read_csv(raw/'health'/'gp_practices_nhs_2026_10.csv', dtype=str)
    gp = gp[(gp['Status Code'] == 'ACTIVE') & (gp['Prescribing Setting (ODS role code)'] == 'RO76')]
    gp = gp.merge(pc[['postcode', 'easting', 'northing']], left_on='Postcode', right_on='postcode')
    addr_cols = ['Address Line 1', 'Address Line 2', 'Address Line 3', 'Address Line 4', 'Address Line 5', 'Postcode']
    parts.append(pd.DataFrame({
        'type': 'gp', 'subtype': 'gp practice', 'id': gp['Organisation Code'],
        'name': gp['Name'], 'info': gp[addr_cols].apply(lambda r: ', '.join(r.dropna()), axis=1),
        'easting': gp['easting'], 'northing': gp['northing'],
    }))

    # Supermarkets: subtype = chain
    shops = pd.read_csv(raw/'supermarkets'/'supermarkets_geolytix_2026_06.csv')
    parts.append(pd.DataFrame({
        'type': 'supermarket', 'subtype': shops['retailer'], 'id': shops['id'].astype(str),
        'name': shops['store_name'], 'info': shops['retailer'] + ', ' + shops['size_band'],
        'easting': shops['bng_e'], 'northing': shops['bng_n'],
    }))

    # Parks: one row per entrance, id = the park so searches keep its nearest gate
    sites = gpd.read_file(raw/'greenspace'/'greenspace_sites_polygons_2026_03.geojson')
    entrances = gpd.read_file(raw/'greenspace'/'greenspace_entrances_points_2026_03.geojson').to_crs(bng)
    park_types = {'Public Park Or Garden': 'park', 'Play Space': 'playground',
                  'Playing Field': 'playing field', 'Allotments Or Community Growing Spaces': 'allotment'}
    sites = sites[sites['function'].isin(park_types)]
    parks = entrances.merge(sites[['id', 'function', 'distName1']], left_on='refToGSite', right_on='id', suffixes=('_entrance', ''))
    parts.append(pd.DataFrame({
        'type': 'park', 'subtype': parks['function'].map(park_types), 'id': parks['id'],
        'name': parks['distName1'].fillna(parks['function']), 'info': parks['function'] + ' (' + parks['accessType'] + ' entrance)',
        'easting': parks.geometry.x, 'northing': parks.geometry.y,
    }))

    # Listed buildings still in force
    listed = load_zones('planning_listed_buildings')
    listed_grade = 'Grade ' + listed['listed-building-grade'].fillna('unknown')
    centres = listed.geometry.centroid
    parts.append(pd.DataFrame({
        'type': 'listed building', 'subtype': listed_grade, 'id': listed['reference'],
        'name': listed['name'], 'info': listed_grade + ' listed building',
        'easting': centres.x, 'northing': centres.y,
    }))

    # Overture places: chosen categories, confidence >= 0.7, not closed, duplicates removed
    places = pd.concat([pd.read_csv(raw/'places'/f'places_overture_2026_09_part{i}.csv', low_memory=False) for i in (1, 2)],
                       ignore_index=True)
    places = places[places['basic_category'].isin(place_types)
                    & (places['confidence'] >= 0.7)
                    & (places['operating_status'] != 'permanently_closed')]
    xy = gpd.points_from_xy(places['lon'], places['lat'], crs='EPSG:4326').to_crs(bng)
    places = places.assign(easting=xy.x, northing=xy.y, name_key=places['name'].str.lower().str.strip())
    places = places.assign(cell_e=places['easting'] // 100, cell_n=places['northing'] // 100)
    places = places.sort_values('confidence', ascending=False).drop_duplicates(subset=['name_key', 'cell_e', 'cell_n'])
    subtype = places['category'].fillna(places['basic_category']).str.replace('_', ' ')
    info = subtype.str.capitalize() + (', ' + places['address']).fillna('') + (', ' + places['postcode']).fillna('')
    parts.append(pd.DataFrame({
        'type': places['basic_category'].map(place_types), 'subtype': subtype, 'id': places['id'],
        'name': places['name'], 'info': info,
        'easting': places['easting'], 'northing': places['northing'],
    }))

    # Combine, keep points inside London, add lat/lon for map pins
    points = pd.concat(parts, ignore_index=True)
    points['easting'] = points['easting'].round().astype(int)
    points['northing'] = points['northing'].round().astype(int)
    xy = gpd.points_from_xy(points['easting'], points['northing'], crs=bng)
    points = points[xy.within(wards.to_crs(bng).union_all())].reset_index(drop=True)
    lonlat = gpd.points_from_xy(points['easting'], points['northing'], crs=bng).to_crs('EPSG:4326')
    points['lon'] = lonlat.x.round(6)
    points['lat'] = lonlat.y.round(6)
    return points

def build_walk_benchmark(pc, points, n=5000):
    """Place counts within a 10-minute walk of a random sample of London postcodes: the yardstick
    the chatbot uses to say how busy a location is compared with the rest of London."""
    tree = cKDTree(points[['easting', 'northing']].to_numpy())
    sample = pc.sample(n, random_state=42)
    counts = [count_by_type(points, tree, x, y, walk_m) for x, y in zip(sample['easting'], sample['northing'])]
    return pd.DataFrame(counts).reset_index(drop=True)

def main():
    """Build all three tables and save them to data/processed/area."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wards = gpd.read_file(raw/'geography'/'boundaries_wards_2024.geojson')
    lookup = pd.read_csv(raw/'geography'/'lookup_lsoa2021_ward2024_borough.csv')

    pc = build_postcodes()
    profile = build_ward_profile(pc, wards, lookup)
    points = build_points(pc, wards)
    walk_benchmark = build_walk_benchmark(pc, points)

    pc.to_parquet(out_dir/'postcodes.parquet', index=False)
    profile.reset_index().to_parquet(out_dir/'ward_profile.parquet', index=False)
    points.to_parquet(out_dir/'points.parquet', index=False)
    walk_benchmark.to_parquet(out_dir/'walk_benchmark.parquet', index=False)
    print(f'postcodes: {pc.shape}, ward_profile: {profile.shape}, points: {points.shape}, '
          f'walk_benchmark: {walk_benchmark.shape}')


if __name__ == '__main__':
    main()