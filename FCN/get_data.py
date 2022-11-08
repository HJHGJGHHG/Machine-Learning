import numpy as np
import pickle as pkl


def f(input):
    return input[0] ** 2 + input[1] ** 2


if __name__ == "__main__":
    num_points = 100
    x = np.linspace(-5, 5, num_points)
    y = np.linspace(-5, 5, num_points)
    X, Y = np.array(np.meshgrid(x, y))
    inputs = np.dstack((X, Y)).reshape(-1, 2)
    
    outputs = [f(input) for input in inputs]
    ground_truth = [(inputs[i: i + 1].reshape(2, 1), outputs[i]) for i in range(len(inputs))]
    np.random.shuffle(ground_truth)
    pkl.dump(ground_truth, open("ground_truth.pkl", "wb"))
