import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def cost_function(
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    return np.square(prediction - target)


def gradient_descent(
    weights: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    condition = (d := 2 * (prediction - target)) > alpha
    decrease_value = d
    decrease_value[condition] = alpha
    return weights - 2 * decrease_value


def make_prediction(
    input_vector: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2


def train_test_split(
    dataset: np.ndarray,
    target: np.ndarray,
    test_size: float = 0.33,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size should be in range <0, 1>")

    if (dataset_len := len(dataset)) != len(target):
        raise ValueError("Length of dataset and targets doesn't match")

    test_samples = int(dataset_len * test_size)

    indices = np.random.permutation(dataset_len)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    train_x, test_x = dataset[train_indices], dataset[test_indices]
    train_y, test_y = target[train_indices], target[test_indices]

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    weights = np.array([1.45, -0.66])
    bias = np.array([0.0])
    DATASET_SIZE = 2000

    dataset = np.random.rand(DATASET_SIZE, 2)
    dataset = dataset * 100

    row_sums = np.sum(dataset, axis=1)
    target = row_sums < 100

    train_dataset, train_target, test_dataset, test_target = train_test_split(
        dataset, target
    )

    for inp, t in zip(train_dataset, train_target):
        pred = make_prediction(inp, weights, bias)
        weights = gradient_descent(weights, t, pred, alpha=0.1)

    accurate_pred = 0

    for inp, t in zip(test_dataset, test_target):
        pred = make_prediction(inp, weights, bias)
        if round(pred[0]) == t:
            accurate_pred += 1

    with open("results.csv", "w") as f:
        for i in range(0, 200, 10):
            for j in range(0, 200, 10):
                inp = np.array([i, j])
                pred = make_prediction(inp, weights, bias)[0]
                f.write("{};{};{};\n".format(i, j, pred))

    print(accurate_pred / len(test_dataset))

    # while True:
    #     a = float(input("Podaj a: "))
    #     b = float(input("Podaj b: "))
    #
    #     inp = np.array([a, b])
    #     is_bigger = make_prediction(inp, weights, bias)[0]
    #
    #     print(is_bigger)
    #     # if is_bigger:
    #     #     print("{} + {} > 100".format(a, b))
    #     # else:
    #     #     print("{} + {} < 100".format(a, b))
