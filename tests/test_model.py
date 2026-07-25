from london_property.model import (
load_bundle, load_postcodes, find_postcode, build_features, predict_price
)

bundle = load_bundle()
postcodes = load_postcodes()

def test_features_match_model_columns():
    row = find_postcode(postcodes, 'SW1A 1AA')
    X = build_features(bundle, row, 70, 'Flat', 'Freehold', 1960)
    assert list(X.columns) == list(bundle['features'])
    assert len(X) == 1
    assert X.notna().all().all()

def test_prediction_plausible():
    row = find_postcode(postcodes, 'SW1A 1AA')
    price = predict_price(bundle, row, 70, 'Flat', 'Freehold', 1960)
    assert 50000 < price < 10000000

def test_property_type_prediction_change():
    row = find_postcode(postcodes, 'SW1A 1AA')
    flat = predict_price(bundle, row, 70, 'Flat', 'Freehold', 1960)
    detached = predict_price(bundle, row, 70, 'Detached', 'Freehold', 1960)
    assert flat != detached

def test_unknown_postcode():
    assert find_postcode(postcodes, 'ZZ69 69ZZ') is None

