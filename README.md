# test-ml

## Manual Random Forest (Regression)

| Parameter          | Value                             |
| ------------------ | --------------------------------- |
| Features           | `["Lag_1", "MA_5", "Lag_1_MA_5"]` |
| Min Samples        | 10                                |
| Number of Trees    | 30                                |
| Max Depth          | 10                                |
| Number of Features | `sqrt(len(features))`             |
| Company            | RMSE %                            |
| --------------     | -----------------                 |
| Apple              | 10.48%                            |
| Tesla              | 4.66%                             |
| Nvidia             | 59.17%                            |
| JP Morgan          | 19.42%                            |
| Shopify            | 3.11%                             |

## Manual Random Forest (Classification)

| Parameter          | Value                                  |
| ------------------ | -------------------------------------- |
| Threshold          | 5% Change over 5 days                  |
| Features           | `["Lag_1", "Lag_2", "MA_5", "Volume"]` |
| Min Samples        | 10                                     |
| Number of Trees    | 30                                     |
| Max Depth          | 10                                     |
| Number of Features | `sqrt(len(features))`                  |
| Company            | F1 Score (Weighted)                    |
| --------------     | -----------------                      |
| Apple              | 0.56                                   |
| Tesla              | 0.46                                   |
| Nvidia             | 0.34                                   |
| JP Morgan          | 0.18                                   |
| Shopify            | 0.29                                   |

## Sci-kit Random Forest (Regression)

## Sci-kit Random Forest (Classification)
