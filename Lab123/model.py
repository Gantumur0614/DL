import cupy as cp 
import numpy as np 


class Linear_Layer:
    def __init__(self, input_dim, output_dim, alpha=0.1):
        self.alpha = alpha
        self.theta = cp.random.randn(input_dim, output_dim) * cp.sqrt(2. / input_dim)
        self.bias = cp.zeros(output_dim)
   
    def forward_pass(self, X):
        self.X = X
        self.z = cp.dot(X, self.theta) + self.bias
        return self.z
    
    def backprop(self, grad_previous):
        self.grad_a = cp.dot(grad_previous, self.theta.T)
        self.grad_theta = cp.dot(self.X.T, grad_previous)
        self.grad_bias = cp.sum(grad_previous, axis=0)
        return self.grad_a
     
    def applying_sgd(self):
        self.theta -= self.alpha * self.grad_theta
        self.bias -= self.alpha * self.grad_bias


class ReLU:
    def forward_pass(self, X):
        self.X = X 
        return cp.maximum(X, 0)
    
    def backprop(self, grad_previous):
        return grad_previous * (self.X > 0)
    
    def applying_sgd(self):
        pass


class Flatten:
    def forward_pass(self, X):
        self.shape = X.shape 
        return X.reshape(self.shape[0], -1)
    
    def backprop(self, grad):
        return grad.reshape(self.shape)
    
    def applying_sgd(self):
        pass


class ConvLayer:
    def __init__(self, in_channels, out_channels, filter_dim=3, stride=1, pad=1, alpha=0.1):
        self.F = filter_dim
        self.S = stride
        self.P = pad
        self.alpha = alpha

        scale = cp.sqrt(2. / (in_channels * filter_dim * filter_dim))
        self.filters = cp.random.randn(
            out_channels, in_channels, filter_dim, filter_dim) * scale
        
        self.bias = cp.zeros(out_channels)

    def convolving(self, X):
        N, C, H, W = X.shape
        out_h = (H + 2*self.P - self.F) // self.S + 1
        out_w = (W + 2*self.P - self.F) // self.S + 1

        X_padded = cp.pad(X, ((0, 0), (0, 0), (self.P, self.P),
                          (self.P, self.P)), mode='constant')

        col = cp.lib.stride_tricks.as_strided(
            X_padded,
            shape=(N, C, self.F, self.F, out_h, out_w),
            strides=(*X_padded.strides[:2], X_padded.strides[2], X_padded.strides[3],
                     self.S*X_padded.strides[2], self.S*X_padded.strides[3])
        )
        col = col.reshape(N, C * self.F * self.F, out_h * out_w)
        return col, (N, C, H, W, out_h, out_w)

    def forward_pass(self, X):
        if X.ndim == 3:
            X = X[:, None, :, :]

        self.x_col, self.shape_info = self.convolving(X)
        N, C, H, W, out_h, out_w = self.shape_info
        FN = self.filters.shape[0]

        w_col = self.filters.reshape(FN, -1)
        out = cp.matmul(w_col, self.x_col) + self.bias[None, :, None]

        out = out.reshape(N, FN, out_h, out_w)

        self.out_shape = out.shape
        return out

    def backprop(self, grad_prev):
        FN, N, out_h, out_w = grad_prev.shape[1], grad_prev.shape[0], grad_prev.shape[2], grad_prev.shape[3]

        dout_flat = grad_prev.transpose(1, 0, 2, 3).reshape(FN, -1)

        col_flat = self.x_col.transpose(
            1, 0, 2).reshape(self.x_col.shape[1], -1)

        self.grad_filters = cp.dot(
            dout_flat, col_flat.T).reshape(self.filters.shape)

        self.grad_bias = cp.sum(dout_flat, axis=1)

        W_flat = self.filters.reshape(FN, -1)
        dcol_flat = cp.dot(W_flat.T, dout_flat)

        d_col = dcol_flat.reshape(col_flat.shape[0], N, -1).transpose(1, 0, 2)

        N, C, H, W, _, _ = self.shape_info
        H_padded, W_padded = H + 2 * self.P, W + 2 * self.P
        dx_padded = cp.zeros((N, C, H_padded, W_padded))

        d_col_reshaped = d_col.reshape(N, C, self.F, self.F, out_h, out_w)

        for i in range(self.F):
            for j in range(self.F):
                dx_padded[:, :, i:i + out_h*self.S: self.S, j:j + out_w *
                          self.S: self.S] += d_col_reshaped[:, :, i, j, :, :]

        if self.P > 0:
            return dx_padded[:, :, self.P:-self.P, self.P:-self.P]
        return dx_padded

    def applying_sgd(self):
        batch_size = self.shape_info[0]
        self.filters -= (self.alpha * self.grad_filters) / batch_size
        self.bias -= (self.alpha * self.grad_bias) / batch_size


class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool = pool_size
        self.stride = stride

    def forward_pass(self, X):
        N, C, H, W = X.shape
        out_h = (H - self.pool) // self.stride + 1
        out_w = (W - self.pool) // self.stride + 1

        col = cp.lib.stride_tricks.as_strided(
            X,
            shape=(N, C, out_h, out_w, self.pool, self.pool),
            strides=(X.strides[0], X.strides[1],
                     self.stride*X.strides[2], self.stride*X.strides[3],
                     X.strides[2], X.strides[3])
        )

        self.X_shape = X.shape
        self.out_shape = (N, C, out_h, out_w)
        self.col = col

        out = cp.max(col, axis=(4, 5))
        return out

    def backprop(self, grad_prev):
        out_expanded = grad_prev[:, :, :, :, None, None]

        max_vals = cp.max(self.col, axis=(4, 5), keepdims=True)
        mask = (self.col == max_vals)

        dx_col = mask * out_expanded

        dx = cp.zeros(self.X_shape)

        if (self.stride == self.pool and
            self.X_shape[2] % self.pool == 0 and
                self.X_shape[3] % self.pool == 0):

            dx = dx_col.transpose(0, 1, 2, 4, 3, 5).reshape(self.X_shape)

        elif self.stride == self.pool:
            grad_windows = dx_col.transpose(0, 1, 2, 4, 3, 5)

            valid_h = self.out_shape[2] * self.pool
            valid_w = self.out_shape[3] * self.pool

            grad_block = grad_windows.reshape(
                self.X_shape[0], self.X_shape[1], valid_h, valid_w)

            dx[:, :, :valid_h, :valid_w] = grad_block

        else:
            for n in range(self.X_shape[0]):
                for c in range(self.X_shape[1]):
                    for h in range(self.out_shape[2]):
                        for w in range(self.out_shape[3]):
                            h_start, w_start = h * self.stride, w * self.stride
                            dx[n, c, h_start:h_start + self.pool,
                                w_start:w_start + self.pool] += dx_col[n, c, h, w]

        return dx

    def applying_sgd(self):
        pass


class SoftMaxCrossEntropy: 
    def forward(self, logits, targets):
        m = targets.shape[0]
        exps = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        self.probs = exps / cp.sum(exps, axis=1, keepdims=True)

        log = -cp.log(self.probs[cp.arange(m), targets] + 1e-9)
        loss = cp.sum(log) / m 
        self.targets = targets
        return loss
    
    def backprop(self):
        m = self.targets.shape[0]
        grad = self.probs.copy()
        grad[cp.arange(m), self.targets] -= 1
        return grad / m
    

class ConvNET:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

    def update(self):
        for layer in self.layers:
            layer.applying_sgd()




















