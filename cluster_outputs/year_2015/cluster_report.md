# NYC Yellow Taxi Behavioral Clustering Report (2015)

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
| 0 | 3001834 | 60.04 |
| 1 | 328323 | 6.57 |
| 2 | 506271 | 10.13 |
| 3 | 1163572 | 23.27 |

## Highest Tip-Rate Clusters
- Cluster 0: 22.95% tip_rate
- Cluster 1: 21.07% tip_rate

## Notes
- Clustering is unsupervised and descriptive
- `tip_rate_pct` is included as a behavioral monetary signal by design
- Cluster patterns are predictive segments, not causal proof