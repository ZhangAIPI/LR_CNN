from utils import get_indices, im2col, col2im
import numpy as np
import torch
import setting

device = setting.device


class Conv():

    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {
            'val': torch.randn(size=(self.n_F, self.n_C, self.f, self.f)).to(device) * np.sqrt(1. / (self.f)),
            'grad': torch.zeros((self.n_F, self.n_C, self.f, self.f)).to(device)}
        self.b = {'val': torch.randn(self.n_F, device=device) * np.sqrt(1. / self.n_F),
                  'grad': torch.zeros((self.n_F), device=device)}

        self.cache = None
        self.w_noise_sigma = 0.01 * torch.ones(size=self.W['val'].reshape(self.n_F, -1).shape, device=device)
        self.noise4w = None
        self.noise4b = None
        self.b_noise_sigma = 0.01 * torch.ones(size=(self.n_F,)).to(device)

    def forward(self, X):
        """
            Performs a forward convolution.

            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W['val'].reshape((self.n_F, -1))
        self.noise4w = torch.randn(size=w_col.shape, device=device)
        self.noise4b = torch.randn(size=self.b_noise_sigma.shape, device=device)
        b_col = self.b['val'].reshape((-1, 1))
        # Perform matrix multiplication.
        # out = w_col  @ X_col + b_col
        out = (w_col + self.w_noise_sigma * self.noise4w) @ X_col + b_col + self.noise4b.reshape(
            (-1, 1)) * self.b_noise_sigma.reshape(
            (-1, 1))
        # Reshape back matrix to image.
        out = torch.cat(torch.split(out, m, dim=1)).reshape((m, n_C, n_H, n_W))
        # out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - dout: error from previous layer.

            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        # Compute bias gradient.
        b_grad = dout[:, np.newaxis] * self.noise4b / self.b_noise_sigma
        b_grad = torch.mean(b_grad, dim=0)
        self.b['grad'] = b_grad
        # Reshape dout properly.
        """
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)"""
        # Reshape dw_col into dw.
        w_col_grad = dout[:, np.newaxis, np.newaxis] * self.noise4w / self.w_noise_sigma
        w_col_grad = torch.mean(w_col_grad, dim=0)
        self.W['grad'] = w_col_grad.reshape((self.n_F, self.n_C, self.f, self.f))
        lr = 0.001
        w_noise_sigma_grad = dout[:, np.newaxis, np.newaxis] * (self.noise4w ** 2 - 1) / self.w_noise_sigma
        b_noise_sigma_grad = dout[:, np.newaxis] * (self.noise4b ** 2 - 1) / self.b_noise_sigma
        self.w_noise_sigma -= lr * w_noise_sigma_grad.mean(dim=0)
        self.b_noise_sigma -= lr * b_noise_sigma_grad.mean(dim=0)
        return self.W['grad'], self.b['grad']


class AvgPool():

    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling.
            Parameters:
            - X: Output of activation function.

            Returns:
            - A_pool: X after average pooling layer.
        """
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = torch.mean(X_col, dim=1)
        # Reshape A_pool properly.
        A_pool = torch.cat(torch.split(A_pool, m, dim=1))
        A_pool = A_pool.reshape((m, n_C, n_H, n_W))

        return A_pool

    def backward(self, dout):
        """
            Distributes error through pooling layer.
            Parameters:
            - dout: Previous layer with the error.

            Returns:
            - dX: Conv layer updated with error.
        """
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = torch.repeat_interleave(dout_flatten, self.f * self.f, dim=0)
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dX properly.
        dX = dX.reshape(m, -1)
        dX = torch.cat(torch.split(dX, n_C_prev, dim=1))
        dX = dX.reshape((m, n_C_prev, n_H_prev, n_W_prev))
        return dX


class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': torch.randn(self.row, self.col, device=device) * np.sqrt(1. / self.col), 'grad': 0}
        self.b = {'val': torch.randn(1, self.row).to(device) * np.sqrt(1. / self.row), 'grad': 0}
        self.ones = torch.ones_like(self.b['val']).to(device)
        self.cache = None
        self.noise = None

    def forward(self, fc):
        """
            Performs a forward propagation between 2 fully connected layers.
            Parameters:
            - fc: fully connected layer.

            Returns:
            - A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = fc.mm(self.W['val'].T) + self.b['val']
        self.noise = torch.randn(size=A_fc.shape, device=device)
        A_fc += self.noise
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.
            Parameters:
            - deltaL: error at last layer.

            Returns:
            - new_deltaL: error at current layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        fc = self.cache
        m = fc.shape[0]

        # Compute gradient.
        w_term = self.cache * deltaL[:, np.newaxis]
        w_batch_grad = torch.einsum('ni, nj->ij', w_term, self.noise)
        # self.ones shape: (n_output,)   deltaL shape: (batch_size, )
        ones = torch.ones(size=[len(self.noise), 1], device=device)
        b_term = ones * deltaL[:, np.newaxis]
        b_batch_grad = torch.einsum('ni, nj->ij', b_term, self.noise)
        self.W['grad'] = w_batch_grad.T / len(self.cache)
        self.b['grad'] = b_batch_grad / len(self.cache)

        # Compute error.
        # We still need to multiply new_deltaL by the derivative of the activation
        # function which is done in TanH.backward().
        return self.W['grad'], self.b['grad']


class SGD():

    def __init__(self, lr, params):
        self.lr = lr
        self.params = params

    def update_params(self, grads):
        for key in self.params:
            self.params[key] = self.params[key] - self.lr * grads['d' + key]
        return self.params


class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = torch.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = torch.zeros(self.params[key].shape)

    def update_params(self, grads):

        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key]
            # RMSprop update.
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (
                    grads['d' + key] ** 2)
            # Update parameters.
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (
                    np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)

        return self.params


class TanH():

    def __init__(self, alpha=1.7159):
        self.alpha = alpha
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.
            Parameters:
            - X: input tensor.
        """
        self.cache = X
        return self.alpha * torch.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * (1 - torch.tanh(X) ** 2)


class ReLU():

    def __init__(self):
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.
            Parameters:
            - X: input tensor.
        """
        self.cache = X
        self.cache[X < 0] = 0
        return self.cache

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.
            Parameters:
            - new_deltaL: error previously computed.
        """
        return new_deltaL


class Softmax():

    def __init__(self):
        pass

    def forward(self, X):
        """
            Compute softmax values for each sets of scores in X.
            Parameters:
            - X: input vector.
        """
        e_x = torch.exp(X - torch.max(X))
        return e_x / torch.sum(e_x, dim=1)[:, np.newaxis]

    def backward(self, y_pred, y):
        return y_pred - y


class CrossEntropyLoss():

    def __init__(self):
        pass

    def get(self, y_pred, y):
        """
            Return the negative log likelihood and the error at the last layer.

            Parameters:
            - y_pred: model predictions.
            - y: ground truth labels.
        """
        loss = -torch.sum(y * torch.log(y_pred))
        return loss
