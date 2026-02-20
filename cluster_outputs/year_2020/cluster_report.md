# NYC Yellow Taxi Behavioral Clustering Report (2020)

## Scope
- Sample rows: 5,000,000
- Behavioral feature count: 8
- Selected k: 4
- k selection metric: `manual_k` = 4.0000
- k selection method: `consensus`
- Grid clustering algorithm used: `minibatch`
- Final clustering algorithm used: `minibatch`
- Winsor bounds: 0.003 to 0.997

## Cluster Sizes
| cluster | n_rows | pct_rows |
|---|---|---|
| 0 | 3316182 | 66.32 |
| 1 | 251582 | 5.03 |
| 2 | 465789 | 9.32 |
| 3 | 966447 | 19.33 |

## Highest Tip-Rate Clusters
- Cluster 0: 28.56% tip_rate
- Cluster 2: 27.03% tip_rate

## Notes
- Clustering is unsupervised and descriptive
- `tip_rate_pct` is included as a behavioral monetary signal by design
- Cluster patterns are predictive segments, not causal proof