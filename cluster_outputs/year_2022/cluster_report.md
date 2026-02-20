# NYC Yellow Taxi Behavioral Clustering Report (2022)

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
| 0 | 385852 | 7.72 |
| 1 | 4057815 | 81.16 |
| 2 | 349177 | 6.98 |
| 3 | 207156 | 4.14 |

## Highest Tip-Rate Clusters
- Cluster 1: 26.67% tip_rate
- Cluster 0: 26.41% tip_rate

## Notes
- Clustering is unsupervised and descriptive
- `tip_rate_pct` is included as a behavioral monetary signal by design
- Cluster patterns are predictive segments, not causal proof