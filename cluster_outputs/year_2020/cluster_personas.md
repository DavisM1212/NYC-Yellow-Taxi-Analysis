# NYC Yellow Taxi Cluster Personas (2020)

Sample rows: 5,000,000

## Tip-Ranked Persona Overview
| cluster | persona | tip_rate_pct | tip_rank | cluster_share_pct | comment |
|---|---|---|---|---|---|
| 0 | High-Tipping Local Errands | 28.56 | 1 | 66.32 | Highest tip propensity, short-hop urban core |
| 2 | Group-Cab City Movers | 27.03 | 2 | 9.32 | Mid-tier tip propensity, group-passenger segment |
| 1 | Airport Connector Long-Haul | 21.66 | 3 | 5.03 | Mid-tier tip propensity, airport-driven |
| 3 | Lower-Tip Mid-Range Trips | 19.04 | 4 | 19.33 | Lowest tip propensity, mixed-value longer hauls |

## Cluster 0 - High-Tipping Local Errands
**Behavioral signature**
- Fare: $7.81 (0.68x overall)
- Distance/Duration/Speed: 1.42 miles, 8.69 min, 10.48 mph
- Passenger count: 1.16
- Airport share: 0.10%
- Tip rate: 28.56%

**Time pattern**
- Top hour bins: Daytime: 44.63%, Evening Commute: 33.16%, Nightlife: 16.65%
- Top day mix: Thursday: 16.33%, Wednesday: 16.02%
- Top month mix: January: 26.48%, February: 26.39%, March: 12.38%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 98.66%, Unknown: 0.58%
- Dropoff borough mix (top): Manhattan: 98.36%, Brooklyn: 0.69%
- Top pickup zones: Upper East Side South, Upper East Side North, Midtown Center, Midtown East, Lincoln Square East
- Top dropoff zones: Upper East Side North, Upper East Side South, Midtown Center, Murray Hill, Lenox Hill West

**Interpretation**
- Short, dense urban hops with strong tipping norms and high transaction frequency make this cluster especially attractive despite smaller fares

**Actionable driver insight**
- Prioritize dense Manhattan short-hop zones during daytime and evening windows, optimizing turnover rate

## Cluster 2 - Group-Cab City Movers
**Behavioral signature**
- Fare: $9.29 (0.80x overall)
- Distance/Duration/Speed: 1.87 miles, 10.57 min, 10.93 mph
- Passenger count: 4.43
- Airport share: 0.27%
- Tip rate: 27.03%

**Time pattern**
- Top hour bins: Daytime: 40.57%, Evening Commute: 33.07%, Nightlife: 19.19%
- Top day mix: Friday: 15.86%, Saturday: 15.69%
- Top month mix: January: 28.98%, February: 28.69%, March: 12.70%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 98.36%, Queens: 0.55%
- Dropoff borough mix (top): Manhattan: 96.39%, Brooklyn: 1.87%
- Top pickup zones: Upper East Side South, Upper East Side North, Midtown Center, Midtown East, Lincoln Square East
- Top dropoff zones: Upper East Side North, Upper East Side South, Midtown Center, Murray Hill, Upper West Side South

**Interpretation**
- Higher passenger counts suggest shared trips, tourism, and social travel patterns with moderate fare levels and generally solid tipping

**Actionable driver insight**
- Target nightlife, hotel, and entertainment pickup zones where multi-passenger demand is common

## Cluster 1 - Airport Connector Long-Haul
**Behavioral signature**
- Fare: $40.30 (3.48x overall)
- Distance/Duration/Speed: 13.83 miles, 33.40 min, 26.09 mph
- Passenger count: 1.52
- Airport share: 88.99%
- Tip rate: 21.66%

**Time pattern**
- Top hour bins: Daytime: 38.60%, Evening Commute: 32.52%, Nightlife: 21.17%
- Top day mix: Thursday: 15.60%, Monday: 15.47%
- Top month mix: January: 33.94%, February: 31.47%, March: 13.47%

**Spatial footprint**
- Pickup borough mix (top): Queens: 66.27%, Manhattan: 31.70%
- Dropoff borough mix (top): Manhattan: 47.13%, Queens: 33.66%
- Top pickup zones: JFK Airport, LaGuardia Airport, Times Sq/Theatre District, Midtown Center, Midtown East
- Top dropoff zones: LaGuardia Airport, JFK Airport, Times Sq/Theatre District, Midtown East, Midtown Center

**Interpretation**
- Airport-to-city and city-to-airport trips dominate this cluster, with high fares and long distances driving steady but not top tip percentages

**Actionable driver insight**
- Stage near JFK/LGA corridors and Manhattan gateways, prioritize queue positioning and clean airport transitions

## Cluster 3 - Lower-Tip Mid-Range Trips
**Behavioral signature**
- Fare: $18.15 (1.57x overall)
- Distance/Duration/Speed: 4.55 miles, 21.27 min, 13.59 mph
- Passenger count: 1.24
- Airport share: 1.13%
- Tip rate: 19.04%

**Time pattern**
- Top hour bins: Daytime: 39.80%, Evening Commute: 31.55%, Nightlife: 20.43%
- Top day mix: Thursday: 16.86%, Friday: 16.52%
- Top month mix: February: 27.43%, January: 25.96%, March: 13.33%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 95.81%, Queens: 1.98%
- Dropoff borough mix (top): Manhattan: 82.32%, Brooklyn: 10.31%
- Top pickup zones: Penn Station/Madison Sq West, Upper East Side North, Midtown Center, Midtown East, Upper East Side South
- Top dropoff zones: Upper East Side North, TriBeCa/Civic Center, Upper West Side North, Yorkville West, Lenox Hill East

**Interpretation**
- Mid-range city hauls show weaker tip behavior relative to fare and time, indicating lower conversion into generous tipping

**Actionable driver insight**
- Use this cluster for volume fill rather than tip optimization, and focus on service quality nudges to improve tip conversion
