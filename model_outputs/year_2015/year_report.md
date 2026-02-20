# NYC Yellow Taxi Modeling Report (2015)

## Scope
- Target: `tip_amount`
- Sample rows: 250,000
- Train rows: 200,000
- Validation rows: 50,000
- Train tip share: 96.51%
- Validation tip share: 96.51%

## Best Models
- Regression winner: `optimized_lightgbm_regression` | RMSE=1.1079 | MAE=0.5860 | R2=0.7483
- Classification winner: `optimized_random_forest_classifier` | ROC-AUC=0.5969 | AP=0.9736 | F1=0.9823

## Baseline Gains
- Regression RMSE gain vs naive mean: 1.1003
- Classification AUC gain vs naive majority: 0.0969

## Cleaning Summary
- Outlier method: `winsor`
- Outlier columns used (6): ['trip_distance', 'duration_min', 'speed_mph', 'fare_amount', 'tolls_amount', 'extra']
- Train rows removed by outlier step: 0
- Validation rows removed by outlier step: 0

## Interpretability Highlights
### Simple Linear Regression (Top Positive Coefficients)
- `is_airport_trip`: 0.6722
- `tolls_amount`: 0.2050
- `fare_amount`: 0.1679
- `extra`: 0.1469
- `month_December`: 0.0595

### Simple Logistic Regression (Top Positive Log-Odds Drivers)
- `borough_Manhattan`: 0.9364
- `borough_Unknown`: 0.8598
- `mta_tax`: 0.7913
- `is_airport_trip`: 0.7106
- `borough_Brooklyn`: 0.4873

## Run Configuration
```json
{
  "year": 2015,
  "target": "tip_amount",
  "target_rows": 250000,
  "seed": 42,
  "valid_frac": 0.2,
  "outlier_method": "winsor",
  "outlier_lower": 0.003,
  "outlier_upper": 0.997,
  "tune_lgbm": true,
  "lgbm_optuna_trials": 40,
  "lgbm_optuna_timeout": null,
  "tune_rf": true,
  "rf_optuna_trials": 30,
  "rf_optuna_timeout": null,
  "rf_max_tune_rows": 150000,
  "plot_crop_quantiles": [
    0.01,
    0.99
  ],
  "plot_crop_padding_frac": 0.03
}
```

## Caveats
- Results are observational and support predictive guidance, not causal claims
- Performance may vary with macro shifts and policy changes across years
- Review subgroup errors before using model outputs for operational recommendations