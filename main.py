import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def cost_function(
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    return np.square(prediction - target)


def gradient_descent(
    weights: np.ndarray, target: np.ndarray, prediction: np.ndarray, alpha: float = 0.01
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


if __name__ == "__main__":
    # Liczba wieksza od 1 Daje 1
    weights = np.array([1.45, -0.66])
    bias = np.array([0.0])

    for _ in range(100000):
        input_vector = np.random.rand(1, 2)[0]
        input_vector = input_vector * 10 + 1

        target = int((input_vector[0] + input_vector[1]) > 5)
        prediction = make_prediction(input_vector, weights, bias)
        weights = gradient_descent(weights, np.array([target]), prediction, alpha=0.1)

    input = np.array([1, 2.63])
    print("0 + 0.63 > 1  => {}".format(make_prediction(input, weights, bias)))
    input = np.array([1.2, 15.63])
    print("1.2 + 0.63 > 1  => {}".format(make_prediction(input, weights, bias)))

# https://realpython.com/python-ai-neural-network/
