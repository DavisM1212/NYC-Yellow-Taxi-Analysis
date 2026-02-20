# NYC Yellow Taxi Cluster Personas (2015)

Sample rows: 5,000,000

## Tip-Ranked Persona Overview
| cluster | persona | tip_rate_pct | tip_rank | cluster_share_pct | comment |
|---|---|---|---|---|---|
| 0 | High-Tipping Local Errands | 22.95 | 1 | 60.04 | Highest tip propensity, short-hop urban core |
| 1 | Airport Connector Long-Haul | 21.07 | 2 | 6.57 | Mid-tier tip propensity, airport-driven |
| 2 | Group-Cab City Movers | 20.81 | 3 | 10.13 | Mid-tier tip propensity, group-passenger segment |
| 3 | Lower-Tip Mid-Range Trips | 16.00 | 4 | 23.27 | Lowest tip propensity, mixed-value longer hauls |

## Cluster 0 - High-Tipping Local Errands
**Behavioral signature**
- Fare: $8.06 (0.64x overall)
- Distance/Duration/Speed: 1.44 miles, 9.03 min, 10.37 mph
- Passenger count: 1.23
- Airport share: 0.14%
- Tip rate: 22.95%

**Time pattern**
- Top hour bins: Daytime: 39.76%, Evening Commute: 26.30%, Nightlife: 22.89%
- Top day mix: Thursday: 15.17%, Friday: 14.97%
- Top month mix: March: 9.31%, January: 9.06%, February: 8.93%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 96.24%, Unknown: 1.59%
- Dropoff borough mix (top): Manhattan: 95.62%, Brooklyn: 2.07%
- Top pickup zones: Union Sq, Upper East Side South, Murray Hill, East Village, Midtown Center
- Top dropoff zones: Midtown Center, Murray Hill, Union Sq, Upper East Side North, Upper East Side South

**Interpretation**
- Short, dense urban hops with strong tipping norms and high transaction frequency make this cluster especially attractive despite smaller fares

**Actionable driver insight**
- Prioritize dense Manhattan short-hop zones during daytime and evening windows, optimizing turnover rate

## Cluster 1 - Airport Connector Long-Haul
**Behavioral signature**
- Fare: $39.52 (3.15x overall)
- Distance/Duration/Speed: 13.46 miles, 33.92 min, 25.36 mph
- Passenger count: 1.70
- Airport share: 91.00%
- Tip rate: 21.07%

**Time pattern**
- Top hour bins: Daytime: 40.28%, Evening Commute: 27.31%, Nightlife: 21.94%
- Top day mix: Monday: 16.38%, Sunday: 15.97%
- Top month mix: May: 9.08%, October: 8.94%, March: 8.83%

**Spatial footprint**
- Pickup borough mix (top): Queens: 59.81%, Manhattan: 38.29%
- Dropoff borough mix (top): Manhattan: 45.49%, Queens: 39.22%
- Top pickup zones: LaGuardia Airport, JFK Airport, Times Sq/Theatre District, Midtown East, Midtown Center
- Top dropoff zones: LaGuardia Airport, JFK Airport, Midtown East, Times Sq/Theatre District, Murray Hill

**Interpretation**
- Airport-to-city and city-to-airport trips dominate this cluster, with high fares and long distances driving steady but not top tip percentages

**Actionable driver insight**
- Stage near JFK/LGA corridors and Manhattan gateways, prioritize queue positioning and clean airport transitions

## Cluster 2 - Group-Cab City Movers
**Behavioral signature**
- Fare: $10.41 (0.83x overall)
- Distance/Duration/Speed: 2.13 miles, 12.02 min, 11.14 mph
- Passenger count: 5.11
- Airport share: 0.41%
- Tip rate: 20.81%

**Time pattern**
- Top hour bins: Daytime: 38.04%, Evening Commute: 25.38%, Nightlife: 23.60%
- Top day mix: Saturday: 16.16%, Friday: 14.97%
- Top month mix: March: 9.21%, April: 8.96%, February: 8.92%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 96.24%, Brooklyn: 1.80%
- Dropoff borough mix (top): Manhattan: 92.60%, Brooklyn: 4.40%
- Top pickup zones: East Village, Union Sq, Upper East Side South, Murray Hill, Midtown East
- Top dropoff zones: Midtown Center, Murray Hill, Upper East Side North, Union Sq, East Village

**Interpretation**
- Higher passenger counts suggest shared trips, tourism, and social travel patterns with moderate fare levels and generally solid tipping

**Actionable driver insight**
- Target nightlife, hotel, and entertainment pickup zones where multi-passenger demand is common

## Cluster 3 - Lower-Tip Mid-Range Trips
**Behavioral signature**
- Fare: $17.51 (1.39x overall)
- Distance/Duration/Speed: 4.21 miles, 20.78 min, 13.14 mph
- Passenger count: 1.27
- Airport share: 0.87%
- Tip rate: 16.00%

**Time pattern**
- Top hour bins: Daytime: 33.77%, Nightlife: 25.82%, Evening Commute: 23.61%
- Top day mix: Thursday: 15.96%, Saturday: 15.50%
- Top month mix: May: 9.35%, March: 9.21%, April: 9.12%

**Spatial footprint**
- Pickup borough mix (top): Manhattan: 93.44%, Brooklyn: 2.89%
- Dropoff borough mix (top): Manhattan: 80.22%, Brooklyn: 11.67%
- Top pickup zones: East Village, Union Sq, TriBeCa/Civic Center, Clinton East, Penn Station/Madison Sq West
- Top dropoff zones: Midtown Center, Upper East Side North, East Village, Midtown East, TriBeCa/Civic Center

**Interpretation**
- Mid-range city hauls show weaker tip behavior relative to fare and time, indicating lower conversion into generous tipping

**Actionable driver insight**
- Use this cluster for volume fill rather than tip optimization, and focus on service quality nudges to improve tip conversion
