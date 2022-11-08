import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def f(input, params):
    return (torch.sin(input[0]) ** 2 + torch.sin(input[1]) ** 2) * (params[0] ** 2 + params[1] ** 2)


def compute_loss(inputs, params):
    loss = 0
    for input in inputs:
        loss += f(input, params).item()
    return loss


if __name__ == "__main__":
    plt.style.reload_library()
    plt.style.use('science')
    num_points = 50
    iters = 3000
    lr = 1e-2
    eps = 1e-5
    limits = 5
    w_1 = np.linspace(-limits, limits, num_points)
    w_2 = np.linspace(-limits, limits, num_points)
    W_1, W_2 = np.meshgrid(w_1, w_2)
    weights = [t for t in np.array(np.meshgrid(w_1, w_2)).T.reshape(-1, 2)]
    
    params = Variable(torch.FloatTensor([0.9 * limits, 0.9 * limits]), requires_grad=True)
    inputs = [torch.FloatTensor((np.random.rand(2) - 0.5) * 10) for _ in range(20)]
    
    losses = np.array([compute_loss(inputs, torch.FloatTensor(param)) for param in weights]).reshape(num_points,
                                                                                                     num_points)
    params_checkpoints_1, params_checkpoints_2 = [], []
    
    # vanilla GD
    tau_epsilon = 0
    for tau in range(iters):
        grad = 0
        a = params
        for instances, input in enumerate(inputs):
            output = f(input, params)
            output.backward()
            # grad += params.grad
            params = Variable(params - lr * params.grad, requires_grad=True)
        
            if -limits < params[0] < limits and -limits < params[1] < limits:
                params_checkpoints_1.append(params[0].detach().numpy())
                params_checkpoints_2.append(params[1].detach().numpy())
            if compute_loss(inputs, params) < eps:
                break
            tau_epsilon += 1
        if compute_loss(inputs, params) < eps:
            break
        else:
            # params = Variable(params - lr * grad / (instances + 1), requires_grad=True)
            pass
    
    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(8, 8))
        plt.title(r"Stochastic Gradient Descent, $\tau(\epsilon)={}$".format(tau_epsilon))
        ct = plt.contour(W_1, W_2, losses, 20, alpha=0.75, cmap=plt.cm.hot_r)
        plt.clabel(ct, inline=True, fontsize=10)  # 添加标签
        plt.plot(params_checkpoints_1, params_checkpoints_2, "bo--")
    
    plt.show()
    print(params)
