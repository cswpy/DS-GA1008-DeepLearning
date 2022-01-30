import torch
import numpy as np

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer, equivalent to linear_1_out_features
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features)
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.cache['x'] = x
        self.parameters['batch_size'] = x.shape[0]
        z1 = x @ self.parameters['W1'].t() + self.parameters['b1'].repeat(self.parameters['batch_size'], 1)
        self.cache['z1'] = z1
        if self.f_function == 'relu':
            self._f_function = torch.nn.ReLU()
            #z2 = torch.nn.ReLU(z1)
        elif self.f_function == 'sigmoid':
            self._f_function = torch.nn.Sigmoid()
            #z2 = torch.nn.Sigmoid(z1)
        else:
            self._f_function = torch.nn.Identity()
        # Shape of z2: (batch_size, linear_1_out_features)
        z2 = self._f_function(z1)
        self.cache['z2'] = z2
        z3 = z2 @ self.parameters['W2'].t() + self.parameters['b2'].repeat(self.parameters['batch_size'], 1)
        self.cache['z3'] = z3
        # Shape of z3: (batch_size, linear_2_out_features)
        if self.g_function == 'relu':
            self._g_function = torch.nn.ReLU()
            #y_hat = torch.nn.ReLU(z3)
        elif self.g_function == 'sigmoid':
            self._g_function = torch.nn.Sigmoid()
            #y_hat = torch.nn.Sigmoid(z3)
        else:
            self._g_function = torch.nn.Identity()
        return self._g_function(z3) # y_hat


    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        # element-wise operation
        if self.g_function == 'relu':
            dJdz3 = dJdy_hat * (self.cache['z3'] > 0) 
        elif self.g_function == 'sigmoid':
            dJdz3 = dJdy_hat * np.exp(-self.cache['z3']) / (1 + np.exp(-self.cache['z3']))**2
        else:
            dJdz3 = dJdy_hat
        # Shape of dJdz3: (batch_size, linear_2_out_features)
        self.grads['dJdW2'] = dJdz3.t() @ self.cache['z2'] # (linear_2_out_features, linear_2_in_features) = (linear_2_out_features, batch_size) @ (batch_size, linear_1_out_features)
        self.grads['dJdb2'] = torch.sum(dJdz3.t(), dim=1) #dJdz3.t() @ torch.ones((self.parameters['batch_size'], 1)) does not work, why?

        dJdz2 = dJdz3 @ self.parameters['W2'] # Shape of dJdz2: (batch_size, linear_2_in_features) = (batch_size, linear_2_out_features) * (linear_2_out_features, linear_2_in_features)
        if self.f_function == 'relu':
            dJdz1 = dJdz2 * (self.cache['z1'] > 0) 
        elif self.f_function == 'sigmoid':
            dJdz1 = dJdz2 * np.exp(-self.cache['z1']) / (1 + np.exp(-self.cache['z1']))**2
        else:
            dJdz1 = dJdz2
        # Shape of dJdz1: (batch_size, linear_1_out_features)
        self.grads['dJdW1'] = dJdz1.t() @ self.cache['x'] # (linear_1_out_features, linear_1_in_features)
        self.grads['dJdb1'] = torch.sum(dJdz1.t(), dim=1) # dJdz1.t() @ torch.ones((self.parameters['batch_size'], 1)) does not work, why? need to flatten

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.mean(torch.square(torch.subtract(y_hat, y)))
    dJdy_hat = 2 / (y.shape[0] * y.shape[1]) * torch.subtract(y_hat, y)

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = torch.mean(-(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))
    dJdy_hat = - 1 / (y.shape[0] * y.shape[1]) * (torch.subtract(y_hat, y) / ((y_hat - 1) * y_hat))
    return loss, dJdy_hat









