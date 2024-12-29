import numpy as np
from auto_random_forest import init_data
from joblib import Parallel, delayed
from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report

config = {"mode": "regression"}


def classify_change(pct_change, buy_threshold=3.0, sell_threshold=-3.0):
    if pct_change > buy_threshold:
        return "Buy"
    elif pct_change < sell_threshold:
        return "Sell"
    else:
        return "Hold"


def bootstrap_sample(data: DataFrame):
    return data.sample(frac=1, replace=True)


def calculate_split_mse(data_left, data_right, target_column, mode=config["mode"]):
    total = len(data_left) + len(data_right)

    if len(data_left) == 0 or len(data_right) == 0:
        return float("inf")

    if mode == "regression":
        left_mean = data_left[target_column].mean()
        right_mean = data_right[target_column].mean()
        return (
            (len(data_left) / total)
            * mean_squared_error(data_left[target_column], left_mean)
        ) + (
            (len(data_right) / total)
            * mean_squared_error(data_right[target_column], right_mean)
        )
    elif mode == "classification":
        return (len(data_left) / total) * calculate_gini(data_left, target_column) + (
            len(data_right) / total
        ) * calculate_gini(data_right, target_column)


def calculate_gini(data, target_column):
    total = len(data)
    if total == 0:  # Avoid division by zero
        return 0
    class_counts = data[target_column].value_counts()
    gini = 1 - sum((count / total) ** 2 for count in class_counts)
    return gini


def calculate_best_split(data: DataFrame, features, target_column, mode=config["mode"]):
    best_mse = float("inf")
    best_feature = None
    best_threshold = None

    for feature in features:
        sorted_data = data.sort_values(feature)
        unique_vals = sorted_data[feature].unique()
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for val in thresholds:
            data_left = sorted_data[sorted_data[feature] < val]
            data_right = sorted_data[sorted_data[feature] >= val]

            if len(data_left) == 0 or len(data_right) == 0:
                continue

            mse = calculate_split_mse(data_left, data_right, target_column, mode=mode)

            if best_mse > mse:
                best_mse = mse
                best_feature = feature
                best_threshold = val

    return best_mse, best_feature, best_threshold


def build_tree(
    data, features, target_column, max_depth, min_samples, depth=0, mode=config["mode"]
):
    if depth >= max_depth:
        return (
            data[target_column].mean()
            if mode == "regression"
            else data[target_column].mode()[0]
        )

    if len(data) < min_samples:
        return (
            data[target_column].mean()
            if mode == "regression"
            else data[target_column].mode()[0]
        )

    # NOTE: check if all values are the same
    if data[target_column].nunique() == 1:
        return (
            data[target_column].mean()
            if mode == "regression"
            else data[target_column].iloc[0]
        )

    best_mse, best_feature, best_threshold = calculate_best_split(
        data, features, target_column, mode=mode
    )

    if not best_feature:
        return (
            data[target_column].mean()
            if mode == "regression"
            else data[target_column].mode()[0]
        )

    data_left = data[data[best_feature] < best_threshold]
    data_right = data[data[best_feature] >= best_threshold]

    left_subtree = build_tree(
        data_left, features, target_column, max_depth, min_samples, depth + 1
    )
    right_subtree = build_tree(
        data_right, features, target_column, max_depth, min_samples, depth + 1
    )

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree,
        "value": (
            data[target_column].mean()
            if mode == "regression"
            else data[target_column].mode()[0]
        ),
    }


def predict_tree(tree, sample, mode=config["mode"]):
    if mode == "regression":
        if not isinstance(tree, dict):
            return tree

        if sample[tree["feature"]] < tree["threshold"]:
            return predict_tree(tree["left"], sample)
        else:
            return predict_tree(tree["right"], sample)
    elif mode == "classification":
        if isinstance(tree, dict):
            if sample[tree["feature"]] < tree["threshold"]:
                return predict_tree(tree["left"], sample)
            else:
                return predict_tree(tree["right"], sample)
        else:
            # Return the majority class stored in the leaf node
            return tree


def train_random_forest(
    data, features, target_column, n_trees, max_depth, min_samples, n_features
):
    trees = []

    for tree in range(n_trees):
        sample = bootstrap_sample(data)

        selected_features = np.random.choice(features, n_features, replace=False)

        tree = build_tree(
            sample, selected_features, target_column, max_depth, min_samples
        )

        trees.append(tree)
    return trees


def train_random_forest_parallel(
    data,
    features,
    target_column,
    n_trees,
    max_depth,
    min_samples,
    n_features,
    mode=config["mode"],
):
    if mode == "regression":
        forest = Parallel(n_jobs=-1)(
            delayed(
                lambda: build_tree(
                    bootstrap_sample(data),
                    np.random.choice(
                        features, n_features, replace=False
                    ),  # Random subset of features
                    target_column,
                    max_depth,
                    min_samples,
                )
            )()
            for _ in range(n_trees)
        )
        return forest
    elif mode == "classification":
        forest = Parallel(n_jobs=-1)(
            delayed(
                lambda: build_tree(
                    bootstrap_sample(data),
                    np.random.choice(
                        features, n_features, replace=False
                    ),  # Random subset of features
                    target_column,
                    max_depth,
                    min_samples,
                )
            )()
            for _ in range(n_trees)
        )
        return forest
    return


def predict_random_forest(forest, sample, mode=config["mode"]):
    predictions = [predict_tree(tree, sample) for tree in forest]

    if mode == "regression":
        return np.mean(predictions)
    elif mode == "classification":
        class_counts = {}
        for pred in predictions:
            class_counts[pred] = class_counts.get(pred, 0) + 1
        # Return the class with the highest count
        return max(class_counts, key=class_counts.get)


def predict_random_forest_batch(forest, data, mode=config["mode"]):
    return data.apply(
        lambda sample: predict_random_forest(forest, sample, mode=mode), axis=1
    )


def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(
        1 for true, pred in zip(true_labels, predicted_labels) if true == pred
    )
    total = len(true_labels)
    return (correct / total) * 100


def get_feature_importances(forest, features):
    feature_importances = {feature: 0 for feature in features}
    for tree in forest:
        if isinstance(tree, dict):
            feature = tree["feature"]
            feature_importances[feature] += 1
    total_splits = sum(feature_importances.values())
    return {k: v / total_splits for k, v in feature_importances.items()}


if __name__ == "__main__":
    training_data, test_data = init_data("./data/shopify_data.csv")

    features = [
        "Lag_1",
        "Lag_2",
        "MA_5",
        "MA_10",
        "SMA_20",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "Volatility_10d",
        "Volume",
    ]

    min_samples = 10
    n_trees = 30
    max_depth = 10
    n_features = int(len(features) ** 0.5)

    if config["mode"] == "regression":
        target_column = "Close"
        print("MODE: REGRESSION")
    elif config["mode"] == "classification":
        target_column = "Signal"
        print("MODE: CLASSIFICATION")
    else:
        exit("config error")

    print("\nTraining Random Forest...")
    forest = train_random_forest_parallel(
        training_data,
        features,
        target_column,
        n_trees,
        max_depth,
        min_samples,
        n_features,
    )
    print(f"Trained {n_trees} trees.")

    print("\nMaking predictions on test data...")
    predictions = predict_random_forest_batch(forest, test_data)

    if config["mode"] == "regression":
        test_data["Predicted_Close"] = predictions
        rmse = (
            mean_squared_error(test_data["Close"], test_data["Predicted_Close"]) ** 0.5
        )
        rmse_percent = (rmse / test_data["Close"].mean()) * 100
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"RMSE %: {rmse_percent}")

    elif config["mode"] == "classification":
        test_data["Predicted_Signal"] = predictions
        accuracy = calculate_accuracy(
            test_data["Signal"], test_data["Predicted_Signal"]
        )
        print(f"Accuracy: {accuracy:.2f}%")
        print(
            classification_report(
                test_data["Signal"],
                test_data["Predicted_Signal"],
                labels=["Buy", "Sell", "Hold"],
            )
        )

    print(get_feature_importances(forest, features))
