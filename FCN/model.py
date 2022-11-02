import tqdm
import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter


def f(input):
    return input[0] + input[1]


class BaseNetwork(object):
    def __init__(self):
        pass
    
    def forward(self, *x):
        pass
    
    def parameters(self):
        pass
    
    def backward(self, grad):
        pass
    
    def __call__(self, *x):
        return self.forward(*x)


class Sequence(BaseNetwork):
    def __init__(self, *layer):
        super(Sequence, self).__init__()
        self.layers = []
        self.parameter = []
        for item in layer:
            self.layers.append(item)
        
        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameter.append(layer.parameters())
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, *x):
        x = x[0]
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def parameters(self):
        return self.parameter
    
    def save_param(self):
        params = []
        for layer in reversed(self.layers):
            if hasattr(layer, "save_param"):
                params.append(layer.save_param())
        return params
    
    def load_param(self, params):
        for param, layer in zip(params, reversed(self.layers)):
            if hasattr(layer, "load_param"):
                layer.load_param(*param)


class Variable(object):
    def __init__(self, weight, wgrad, bias, bgrad, v_weight=None):
        self.weight = weight
        self.wgrad = wgrad
        self.v_weight = np.zeros(self.weight.shape) if v_weight is None else v_weight
        self.bias = bias
        self.bgrad = bgrad
    
    def save_param(self):
        # 参数保存
        return self.weight, self.wgrad, self.v_weight, self.bias, self.bgrad


class Linear(BaseNetwork):
    def __init__(self, input_dim, output_dim, std=0.1):
        super(Linear, self).__init__()
        self.weight = np.random.normal(loc=0.0, scale=std, size=(output_dim, input_dim))
        self.bias = np.random.normal(loc=0.0, scale=std, size=[output_dim, 1])
        self.input, self.output = None, None
        self.wgrad, self.bgrad = np.zeros(self.weight.shape), np.zeros(self.bias.shape)
        self.variable = Variable(self.weight, self.wgrad, self.bias, self.bgrad)
    
    def parameters(self):
        return self.variable
    
    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = np.dot(self.weight, self.input) + self.bias
        return self.output
    
    def backward(self, grad):
        self.bgrad += grad
        self.wgrad += np.dot(grad, self.input.T)
        grad = np.dot(self.weight.T, grad)
        return grad
    
    def save_param(self):
        return self.variable.save_param()
    
    def load_param(self, weight, wgrad, v_weight, bias, bgrad):
        assert self.variable.weight.shape == weight.shape
        assert self.variable.wgrad.shape == wgrad.shape
        assert self.variable.v_weight.shape == v_weight.shape
        assert self.variable.bias.shape == bias.shape
        assert self.variable.bgrad.shape == bgrad.shape
        self.weight = weight
        self.bias = bias
        self.wgrad = wgrad
        self.bgrad = bgrad
        self.variable = Variable(weight, wgrad, bias, bgrad, v_weight)


class ReLU(BaseNetwork):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input, self.output = None, None
    
    def forward(self, *x):
        x = x[0]
        self.input = x
        x[self.input <= 0] *= 0
        self.output = x
        return self.output
    
    def backward(self, grad):
        grad[self.input > 0] *= 1
        grad[self.input <= 0] *= 0
        return grad


class MSE(object):
    def __init__(self):
        self.label, self.pred, self.grad, self.loss = None, None, None, None
    
    def __call__(self, pred, label):
        return self.forward(pred, label)
    
    def forward(self, pred, label):
        self.pred, self.label = pred, label
        self.loss = np.sum(0.5 * np.square(self.pred - self.label))
        return self.loss
    
    def backward(self, grad=None):
        self.grad = (self.pred - self.label)
        ret_grad = np.sum(self.grad, axis=0)
        return np.expand_dims(ret_grad, axis=0)


class Vanilla_GD:
    def __init__(self, parameters, lr=1e-4):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for parameters in self.parameters:
            parameters.wgrad *= 0
            parameters.bgrad *= 0
    
    def step(self):
        for parameters in self.parameters:
            parameters.weight += parameters.v_weight - self.lr * parameters.wgrad
            parameters.bias -= self.lr * parameters.bgrad


class Mynet(BaseNetwork):
    def __init__(self, input_dim, hidden_dim, std=0.1):
        super(Mynet, self).__init__()
        self.layers = Sequence(
            Linear(input_dim, hidden_dim, std=0.1),
            ReLU(),
            Linear(hidden_dim, 1, std=0.1)
        )
        self.criterion = MSE()
    
    def parameters(self):
        return self.layers.parameters()
    
    def forward(self, *x):
        x = x[0]
        return self.layers.forward(x)
    
    def backward(self, grad=None):
        grad = self.criterion.backward(grad)
        self.layers.backward(grad)


def evaluate(model, test_dataset, loss_func):
    losses = []
    for input, target in test_dataset:
        output = model(input)
        loss = loss_func(output, target)
        losses.append(loss)
    return float(np.mean(losses))


def train(model, train_dataset, test_dataset, optimizer, epochs):
    losses = 0
    step = 1
    writer = SummaryWriter("/root/tf-logs/1")
    loss_func = model.criterion
    for epoch in tqdm.tqdm(range(epochs)):
        for instances, data in enumerate(train_dataset):
            output = model(data[0])
            loss = loss_func(output, data[1])
            
            model.backward()
            
            losses += loss
            step += 1
            # writer.add_scalars("Loss", {"train_loss": loss}, step)
        
        writer.add_scalars("Loss", {"train_loss": losses / (instances + 1)}, epoch + 1)
        
        optimizer.step()
        optimizer.zero_grad()
        
        test_loss = evaluate(model, test_dataset, loss_func)
        writer.add_scalars("Loss", {"test_loss": test_loss}, epoch + 1)
        losses, instances = 0, 0
        if (epoch + 1) % 20 == 0:
            print("Epoch {0}, Avg MSE Loss: {1:6f}".format(epoch + 1, losses / (instances + 1)))
            print("Avg Test_dataset MSE Loss: {0:6f}".format(test_loss))
    
    writer.close()


if __name__ == "__main__":
    data = pkl.load(open("ground_truth.pkl", "rb"))
    np.random.shuffle(data)
    train_dataset = data[300:]
    test_dataset = data[:300]
    
    hidden_dim = 100
    lr = 1e-6
    epochs = 50
    model = Mynet(input_dim=2, hidden_dim=100)
    vanilla_model = model
    loss_func = model.criterion
    optimizer = Vanilla_GD(model.parameters(), lr=lr)
    
    # 可视化
    plt.style.reload_library()
    plt.style.use('science')
    num_points = 1000
    
    x = np.linspace(-5, 5, num_points)
    y = np.linspace(-5, 5, num_points)
    X, Y = np.array(np.meshgrid(x, y))
    inputs = [t for t in np.array(np.meshgrid(x, y)).T.reshape(-1, 2, 1)]
    
    Z = np.array(list(map(f, inputs))).reshape(num_points, num_points)
    Z_1 = np.array(
        [model.forward(input) for input in inputs]
    ).reshape(num_points, num_points)
    
    # 训练
    train(model, train_dataset, test_dataset, optimizer, epochs)
    
    Z_2 = np.array(
        [model.forward(input) for input in inputs]
    ).reshape(num_points, num_points)
    
    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='YlOrRd', edgecolor='none')
        # ax.plot_surface(X, Y, Z_1, cmap='binary', edgecolor='none')
        # ax.plot_surface(X, Y, Z_2, cmap='viridis', edgecolor='none')
        ax.plot_surface(X, Y, Z_2, cmap='binary', edgecolor='none')
        ax.set_xlabel(u'$x_1$')
        ax.set_ylabel(u'$x_2$')
        ax.set_zlabel(u'$f(x_1,x_2)$')
    
    plt.savefig("/root/autodl-tmp/Machine-Learning/FCN/2.png")
    plt.show()
