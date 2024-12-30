# test-ml

## Manual Random Forest (Regression)

| Parameter          | Value                             |
| ------------------ | --------------------------------- |
| Features           | `["Lag_1", "MA_5", "Lag_1_MA_5"]` |
| Min Samples        | 10                                |
| Number of Trees    | 30                                |
| Max Depth          | 10                                |
| Number of Features | `sqrt(len(features))`             |

| Company   | RMSE % |
| --------- | ------ |
| Apple     | 12.95% |
| Tesla     | 17.84% |
| Nvidia    | 65.14% |
| JP Morgan | 19.42% |
| Shopify   | 4.55%  |

## Manual Random Forest (Classification)

| Parameter          | Value                                                   |
| ------------------ | ------------------------------------------------------- |
| Features           | `Lag_1`, `Lag_2`, `MA_5`, `MA_10`, `SMA_20`,            |
|                    | `Bollinger_Upper`, `Bollinger_Lower`, `Volatility_10d`, |
|                    | `Volume`                                                |
| Min Samples        | 10                                                      |
| Number of Trees    | 30                                                      |
| Max Depth          | 10                                                      |
| Number of Features | `sqrt(len(features))`                                   |

| Company   | F1 Score (Weighted) |
| --------- | ------------------- |
| Apple     | 0.56                |
| Tesla     | 0.46                |
| Nvidia    | 0.34                |
| JP Morgan | 0.64                |
| Shopify   | 0.50                |

## Sci-kit Random Forest (Regression)

| Company   | RMSE % |
| --------- | ------ |
| Apple     | 11.65% |
| Tesla     | 4.06%  |
| Nvidia    | 59.02% |
| JP Morgan | 19.50% |
| Shopify   | 0.59%  |

## Sci-kit Random Forest (Classification)

| Company   | F1 Score (Weighted) |
| --------- | ------------------- |
| Apple     | 0.61                |
| Tesla     | 0.61                |
| Nvidia    | 0.34                |
| JP Morgan | 0.62                |
| Shopify   | 0.52                |
