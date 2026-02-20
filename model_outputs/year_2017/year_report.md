# NYC Yellow Taxi Modeling Report (2017)

## Scope
- Target: `tip_amount`
- Sample rows: 250,000
- Train rows: 200,000
- Validation rows: 50,000
- Train tip share: 95.91%
- Validation tip share: 95.90%

## Best Models
- Regression winner: `optimized_lightgbm_regression` | RMSE=1.1238 | MAE=0.5473 | R2=0.7526
- Classification winner: `optimized_random_forest_classifier` | ROC-AUC=0.5948 | AP=0.9680 | F1=0.9791

## Baseline Gains
- Regression RMSE gain vs naive mean: 1.1358
- Classification AUC gain vs naive majority: 0.0948

## Cleaning Summary
- Outlier method: `winsor`
- Outlier columns used (6): ['trip_distance', 'duration_min', 'speed_mph', 'fare_amount', 'tolls_amount', 'extra']
- Train rows removed by outlier step: 0
- Validation rows removed by outlier step: 0

## Interpretability Highlights
### Simple Linear Regression (Top Positive Coefficients)
- `is_airport_trip`: 0.5435
- `tolls_amount`: 0.2025
- `fare_amount`: 0.1748
- `extra`: 0.1412
- `month_April`: 0.0355

### Simple Logistic Regression (Top Positive Log-Odds Drivers)
- `is_airport_trip`: 0.8846
- `borough_Unknown`: 0.8643
- `borough_Manhattan`: 0.8489
- `hour_Evening Commute`: 0.4788
- `hour_Nightlife`: 0.4517

## Run Configuration
```json
{
  "year": 2017,
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