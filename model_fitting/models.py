"""
@author: Navid
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Set the default tensor type to float64
# torch.set_default_dtype(torch.float64)

#############################
#     Pure torch models     #
#############################

# --- Multilayer Perceptron ---#
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        extend_states=False,
        activation=None,
        input_normalizer_params=None,
        output_normalizer_params=None,
        device = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.extend_states = extend_states
        self.input_normalizer_params = input_normalizer_params
        self.output_normalizer_params = output_normalizer_params
        
        assert num_layers >= 1, "There must be at least one hidden layer"
        # input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # hidden layers
        hidden = []
        i = -1
        for i in range(num_layers - 1):
            hidden.append((nn.Linear(hidden_size, hidden_size)))
        self.hidden_layers = torch.nn.ModuleList(hidden)
        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            # self.act = getattr(nn.activation, activation)()
            self.act = nn.ReLU()
        else:
            self.act = activation

        if input_normalizer_params is not None:
            self.x_min = torch.tensor(input_normalizer_params["params_x"]["min"], dtype=torch.float32)
            self.x_max = torch.tensor(input_normalizer_params["params_x"]["max"], dtype=torch.float32)
        
        if output_normalizer_params is not None:
            self.y_min = torch.tensor(output_normalizer_params["params_x"]["min"], dtype=torch.float32)
            self.y_max = torch.tensor(output_normalizer_params["params_x"]["max"], dtype=torch.float32)

    def forward(self, x):
        # normalize the input if necessary (for encoder)
        if self.input_normalizer_params is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        # itterdate over layers
        x = x.reshape(x.size(0), -1)
        h = self.input_layer(x)
        h = self.act(h)
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.act(h)
        y = self.output_layer(h)
        if self.extend_states:
            y = torch.concat((x, y), axis=1)
        # de normalize the input if necessary (for decoder)
        if self.output_normalizer_params is not None:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y

    def get_inputsize(self):
        return self.input_size

# --- Multilayer Perceptron ---#
class DoubleMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation=None,
        input_normalizer_params=None,
        output_normalizer_params=None,
        device = None
    ):
        super().__init__()
        self.input_size = int(input_size/2)
        self.hidden_sizes = [int(x/2) for x in hidden_sizes]
        self.output_size = int(output_size/2)
        self.input_normalizer_params = input_normalizer_params
        self.output_normalizer_params = output_normalizer_params

        # MLP1 = MLP(int(input_size/2), int(hidden_sizes/2), int(output_size/2), activation, input_normalizer_params,  output_normalizer_params, device)

        assert len(hidden_sizes) >= 1, "There must be at least one hidden layer"
        # input layer
        self.input_layer_1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.input_layer_2 = nn.Linear(self.input_size, self.hidden_sizes[0])
        # hidden layers
        hidden_1 = []
        hidden_2 = []
        i = -1
        for i in range(len(hidden_sizes) - 1):
            hidden_1.append((nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])))
            hidden_2.append((nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])))
        self.hidden_layers_1 = torch.nn.ModuleList(hidden_1)
        self.hidden_layers_2 = torch.nn.ModuleList(hidden_2)
        # output layer
        self.output_layer_1 = nn.Linear(self.hidden_sizes[i + 1], self.output_size)
        self.output_layer_2 = nn.Linear(self.hidden_sizes[i + 1], self.output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            # self.act = getattr(nn.activation, activation)()
            self.act = nn.ReLU()
        else:
            self.act = activation

        if input_normalizer_params is not None:
            self.x_min = torch.tensor(input_normalizer_params["params_x"]["min"], dtype=torch.float32)
            self.x_max = torch.tensor(input_normalizer_params["params_x"]["max"], dtype=torch.float32)
        
        if output_normalizer_params is not None:
            self.y_min = torch.tensor(output_normalizer_params["params_y"]["min"], dtype=torch.float32)
            self.y_max = torch.tensor(output_normalizer_params["params_y"]["max"], dtype=torch.float32)

    def forward(self, x):
        # normalize the input if necessary
        if self.input_normalizer_params is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        # separate the inputs vectors to inputs of each MLP
        x_1 = x[:,0::2]
        x_2 = x[:,1::2]
        # itterdate over layers _ MLP1
        x_1 = self.input_layer_1(x_1)
        x_1 = self.act(x_1)
        for layer in self.hidden_layers_1:
            x_1 = layer(x_1)
            x_1 = self.act(x_1)
        y_1 = self.output_layer_1(x_1)
        # itterdate over layers _ MLP2
        x_2 = self.input_layer_2(x_2)
        x_2 = self.act(x_2)
        for layer in self.hidden_layers_2:
            x_2 = layer(x_2)
            x_2 = self.act(x_2)
        y_2 = self.output_layer_2(x_2)
        y = torch.cat([torch.cat((y_1[:, i:i+1], y_2[:, i:i+1]), dim=1) for i in range(y_1.shape[1])], dim=1)
        # de normalize the input if necessary
        if self.output_normalizer_params is not None:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y

    def get_inputsize(self):
        return self.input_size *2 

# --- Multilayer Perceptron with 2D delayed nput---#
class DelayedMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        len_history,
        output_size,
        activation=None,
        input_normalizer_params=None,
        output_normalizer_params=None,
        device = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_normalizer_params = input_normalizer_params
        self.output_normalizer_params = output_normalizer_params
        
        assert num_layers >= 1, "There must be at least one hidden layer"
        # input layer
        self.input_layer = nn.Linear(input_size*len_history, hidden_size)
        # hidden layers
        hidden = []
        i = -1
        for i in range(num_layers - 1):
            hidden.append((nn.Linear(hidden_size, hidden_size)))
        self.hidden_layers = torch.nn.ModuleList(hidden)
        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            # self.act = getattr(nn.activation, activation)()
            self.act = nn.ReLU()
        else:
            self.act = activation

        if input_normalizer_params is not None:
            self.x_min = torch.tensor(input_normalizer_params["params_x"]["min"], dtype=torch.float32)
            self.x_max = torch.tensor(input_normalizer_params["params_x"]["max"], dtype=torch.float32)
        
        if output_normalizer_params is not None:
            self.y_min = torch.tensor(output_normalizer_params["params_x"]["min"], dtype=torch.float32)
            self.y_max = torch.tensor(output_normalizer_params["params_x"]["max"], dtype=torch.float32)

    def forward(self, x):
        # normalize the input if necessary
        if self.input_normalizer_params is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        # itterdate over layers
        x = self.input_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        y = self.output_layer(x)
        # de normalize the input if necessary
        if self.output_normalizer_params is not None:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y

    def get_inputsize(self):
        return self.input_size

# --- Long Short-Term Memory Network ---#
class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        activation=None,
        input_normalizer_params=None,
        output_normalizer_params=None,
        device=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_normalizer_params = input_normalizer_params
        self.output_normalizer_params = output_normalizer_params
        
        assert hidden_size >= 1, "There must be at least one hidden layer"
        # input LSTM layer
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # output linear layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            self.act = nn.ReLU()
        else:
            self.act = activation

        if input_normalizer_params is not None:
            self.x_min = torch.tensor(input_normalizer_params["params_x"]["min"], dtype=torch.float32, device=device)
            self.x_max = torch.tensor(input_normalizer_params["params_x"]["max"], dtype=torch.float32, device=device)
        
        if output_normalizer_params is not None:
            self.y_min = torch.tensor(output_normalizer_params["params_x"]["min"], dtype=torch.float32, device=device)
            self.y_max = torch.tensor(output_normalizer_params["params_x"]["max"], dtype=torch.float32, device=device)

    def forward(self, x):
        # Normalize the input if necessary
        if self.input_normalizer_params is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # Processing the input through the input LSTM layer
        x, (hn, cn) = self.lstm_layer(x)

        # Passing the output of the last LSTM layer through the output linear layer
        y = self.output_layer(x[:, -1, :])  # Assuming you want the last time step for prediction
        
        # Denormalize the output if necessary
        if self.output_normalizer_params is not None:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y

# --- Linear Koopman Operator (just a constant A matrix) ---#
class TrainableA(nn.Module):
    def __init__(
        self,
        states_dim,
    ):
        super().__init__()
        self.dimension = states_dim
        self.linear_model_layer = nn.Linear(states_dim, states_dim, bias=False)

    def forward(self, x):
        y = self.linear_model_layer(x)
        return y

# --- Linear Koopman - actuation input part (just a constant B matrix) ---#
class TrainableB(nn.Module):
    def __init__(
        self,
        states_dim,
        inputs_dim
    ):
        super().__init__()
        self.linear_model_layer = nn.Linear(inputs_dim, states_dim, bias=False)

    def forward(self, u):
        y = self.linear_model_layer(u)
        return y

# --- Full model including encoder, constant Koopman operator, and decoder ---#
class CombinedDeepKoopman(nn.Module):
    def __init__(self, encoder, states_matrix, actuation_matrix, decoder):
        super().__init__()
        self.encoder = encoder
        self.states_matrix = states_matrix
        self.actuation_matrix = actuation_matrix
        self.decoder = decoder

    def forward(self, x):
        y = self.encoder(x)
        y_plus = self.states_matrix(y)[0] + self.actuation_matrix(y)[0]
        x_plus = self.decoder(y_plus)
        return y, y_plus, x_plus
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        self.states_matrix.requires_grad_(status)
        self.actuation_matrix.requires_grad_(status)
        self.decoder.requires_grad_(status)

# --- Full model including encoder, constant Koopman operator, and decoder ---#
class CombinedDeepKoopman_without_decoder(nn.Module):
    def __init__(self, encoder, states_matrix, actuation_matrix):
        super().__init__()
        self.encoder = encoder
        self.states_matrix = states_matrix
        self.actuation_matrix = actuation_matrix

    def forward(self, x):
        y = self.encoder(x)
        y_plus = self.states_matrix(y)[0] + self.actuation_matrix(y)[0]
        x_plus = y_plus[0:self.encoder.get_inputsize()]
        return y, y_plus, x_plus
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        self.states_matrix.requires_grad_(status)
        self.actuation_matrix.requires_grad_(status)


# --- An auxiliary network for predicting Eigen values of the Koopman operator ---#
class States_Auxiliary(nn.Module):
    def __init__(self, num_complexeigens_pairs, num_realeigens, hidden_sizes, activation=None, device=None):
        super().__init__()
        self.num_complexeigens_pairs = num_complexeigens_pairs
        self.num_realeigens = num_realeigens
        self.net = nn.ModuleList()
        for i in range(num_complexeigens_pairs):
            self.net.append(MLP(input_size=1, hidden_sizes=hidden_sizes, output_size=2, activation=activation))

        for i in range(num_realeigens):
            self.net.append(MLP(input_size=1, hidden_sizes=hidden_sizes, output_size=1, activation=activation))

    def forward(self, y):
        Lambdas = []
        yblock = y[:, 0*2:0*2+2]
        # yradius = torch.sqrt(torch.mean(yblock**2, dim=1)).unsqueeze(dim=1)
        yradius = torch.sqrt(torch.sum(yblock**2, dim=1)).unsqueeze(dim=1)
        Lambdas.append(self.net[0](yradius))

        # for i in range(self.num_complexeigens_pairs):
        #     yblock = y[:, i*2:i*2+2]
        #     # yradius = torch.sqrt(torch.mean(yblock**2, dim=1)).unsqueeze(dim=1)
        #     yradius = torch.sqrt(torch.sum(yblock**2, dim=1)).unsqueeze(dim=1)
        #     Lambdas.append(self.net[i](yradius))

        # for i in range(self.num_realeigens):
        #     yblock = y[:, self.num_complexeigens_pairs*2 + i].unsqueeze(dim=1) 
        #     Lambdas.append(self.net[self.num_complexeigens_pairs+i](yblock))
            
        Lambdas = torch.cat(Lambdas, dim=1)  # Use torch.cat for proper tensor concatenation
        return  Lambdas

# --- Universal space varying (Eigen values varying) Koopman Operator based on Bethany Lusch Nature paper ---#
class VariableKoopman(nn.Module):
    def __init__(self, num_complexeigens_pairs, num_realeigens, sample_time, structure="Jordan"):
        """
        Initialize the module with n complex Jordan blocks and m real eigenvalues.
        Dimension d = 2*n + m, since each complex Jordan block is 2-dimensional.

        Args:
        n (int): Number of complex Jordan blocks.
        m (int): Number of real eigenvalues.
        dt (float): Time step delta t.
        """
        super(VariableKoopman, self).__init__()
        self.num_complexeigens_pairs = num_complexeigens_pairs  # num Complex Pairs of Eigen values
        self.num_realeigens = num_realeigens  # num Real Eigen values
        self.sample_time = sample_time
        self.dimension = 2*num_complexeigens_pairs + num_realeigens
        self.structure = structure
        

    def forward(self, y, Lambda):
        self.batch_size = Lambda.shape[0]
        self.device = Lambda.device

        # Initialize A as a batch of identity matrices
        self.G = torch.eye(self.dimension, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.A = torch.eye(self.dimension, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        # # Update mu, omega, and real_eig based on Lambda
        # mu = Lambda[:, 0:self.num_complexeigens_pairs]  
        # omega = Lambda[:, self.num_complexeigens_pairs:2*self.num_complexeigens_pairs]
        # real_eig = Lambda[:, 2*self.num_complexeigens_pairs:]

        # 
        if self.structure == "Jordan" : 
            G = self.update_real_canonical_disc(Lambda, self.sample_time)
            A = self.update_real_canonical_cont(Lambda)
            # G = torch.linalg.matrix_exp(A*self.sample_time)
        elif self.structure == "Controlable" : 
            A = self.update_cont_canonical_cont(Lambda)
            G = torch.linalg.matrix_exp(A*self.sample_time)
        else:
            raise ValueError("Unknown structure for A matrix")


        # Compute y = A * x for each item in the batch
        y = y.unsqueeze(-1)  # Add an extra dimension for matmul
        x_plus = torch.matmul(G, y).squeeze(-1)  # Remove the extra dimension after matmul

        return x_plus, G, A
    
    def update_real_canonical_disc(self, Lambda, sample_time):
        # Update mu, omega, and real_eig based on Lambda
        mu = Lambda[:, 0:2*self.num_complexeigens_pairs:2]  
        omega = Lambda[:, 1:2*self.num_complexeigens_pairs:2]
        real_eig = Lambda[:, 2*self.num_complexeigens_pairs:]

        # Update G for complex eigenvalues
        for i in range(self.num_complexeigens_pairs):
            exp_mu_dt = torch.exp(mu[:, i] * sample_time)
            cos_omega_dt = torch.cos(omega[:, i] * sample_time)
            sin_omega_dt = torch.sin(omega[:, i] * sample_time)
            # Constructing the 2x2 block for each item in the batch
            block_disc = torch.stack([
                exp_mu_dt * cos_omega_dt, -exp_mu_dt * sin_omega_dt,
                exp_mu_dt * sin_omega_dt, exp_mu_dt * cos_omega_dt
            ], dim=-1).view(self.batch_size, 2, 2)
            # Placing the block in the correct position for each A in the batch
            self.G[:, 2*i:2*(i+1), 2*i:2*(i+1)] = block_disc

        # Update G for real eigenvalues
        for i in range(self.num_realeigens):
            exp_lambda_dt = torch.exp(real_eig[:, i] * sample_time)
            self.G[:, 2*self.num_complexeigens_pairs + i, 2*self.num_complexeigens_pairs + i] = exp_lambda_dt

        return self.G

    def update_real_canonical_cont(self, Lambda):
        # Update mu, omega, and real_eig based on Lambda
        mu = Lambda[:, 0:2*self.num_complexeigens_pairs:2]  
        omega = Lambda[:, 1:2*self.num_complexeigens_pairs:2]
        real_eig = Lambda[:, 2*self.num_complexeigens_pairs:]

        # Update G for complex eigenvalues
        for i in range(self.num_complexeigens_pairs):
            # Constructing the 2x2 block for each item in the batch
            block_cont = torch.stack([
                mu[:, i], -omega[:, i],
                omega[:, i], mu[:, i]
            ], dim=-1).view(self.batch_size, 2, 2)
            # Placing the block in the correct position for each A in the batch
            self.A[:, 2*i:2*(i+1), 2*i:2*(i+1)] = block_cont

        # Update G for real eigenvalues
        for i in range(self.num_realeigens):
            self.A[:, 2*self.num_complexeigens_pairs + i, 2*self.num_complexeigens_pairs + i] = real_eig[:, i]

        return self.A

    def update_cont_canonical_disc(self, Lambda, sample_time):
        # Update mu, omega, and real_eig based on Lambda
        mu = Lambda[:, 0:2*self.num_complexeigens_pairs:2]  
        omega = Lambda[:, 1:2*self.num_complexeigens_pairs:2]
        real_eig = Lambda[:, 2*self.num_complexeigens_pairs:]

        P_real_to_cont = torch.tensor([[-1, 0], [-mu/omega, 1/omega]], dtype=torch.float32)

        G_r = self.update_real_canonical_disc(Lambda, sample_time)

        G_c = torch.matmul(torch.matmul(torch.inverse(P_real_to_cont), G_r), P_real_to_cont)

        return G_c

    def update_cont_canonical_cont(self, Lambda,):
        # Update mu, omega, and real_eig based on Lambda
        mu = Lambda[:, 0:2*self.num_complexeigens_pairs:2]  
        omega = Lambda[:, 1:2*self.num_complexeigens_pairs:2]
        real_eig = Lambda[:, 2*self.num_complexeigens_pairs:]

        # Update G for complex eigenvalues
        for i in range(self.num_complexeigens_pairs):
            # Constructing the 2x2 block for each item in the batch
            block_cont = torch.stack([
                torch.zeros_like(omega[:, i]), torch.ones_like(omega[:, i]),
                -1*(omega[:, i]**2 + mu[:, i]**2), 2*mu[:, i]
            ], dim=-1).view(self.batch_size, 2, 2)
            # Placing the block in the correct position for each A in the batch
            self.A[:, 2*i:2*(i+1), 2*i:2*(i+1)] = block_cont

        # Update G for real eigenvalues
        for i in range(self.num_realeigens):
            raise ValueError("This part of the code is not developed yet")

        return self.G

# --- Full model including encoder, space varying Koopman operator, and decoder ---#
class CombinedKoopman_withAux_withoutInput(nn.Module):
    def __init__(self, encoder, auxiliary, koopman, decoder):
        super().__init__()
        self.encoder = encoder
        self.states_auxiliary = auxiliary
        self.koopman = koopman
        self.decoder = decoder

    def forward(self, x):
        y = self.encoder(x)
        Lambda = self.states_auxiliary(y)
        y_plus = self.koopman(y, Lambda)
        x_plus = self.decoder(y_plus)
        return y, y_plus, x_plus
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        for i in range(len(self.states_auxiliary.net)):
            self.states_auxiliary.net[i].requires_grad_(status)
        self.decoder.requires_grad_(status)

    def Kmat(self, y, omega):
        w = omega[:,0]
        mu = w = omega[:,1]
        dt = 2E-2
        # Compute exp_term and rotation terms for each element in the batch
        exp_term = torch.exp(mu * dt).view(-1, 1, 1)
        cos_wdt = torch.cos(w * dt).view(-1, 1, 1)
        sin_wdt = torch.sin(w * dt).view(-1, 1, 1)

        # Create a batch of 2x2 rotation matrices
        zero = torch.zeros_like(cos_wdt)
        rot_term = torch.cat([torch.cat([cos_wdt, -sin_wdt], dim=2),
                              torch.cat([sin_wdt, cos_wdt], dim=2)], dim=1)
        
        B = exp_term * rot_term

        # Reshape y for batch matrix multiplication
        y_reshaped = y.view(y.shape[0], 2, -1) # Assuming y has shape [batch_size, 2 * n]

        y_plus = torch.matmul(B, y_reshaped)

        # Reshape y_plus back to original shape if necessary
        y_plus = y_plus.view(y.shape)
        return y_plus

# --- An auxiliary network for predicting B matrix for the input term ---#    
class Input_Auxiliary(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_shape, activation=None, device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_shape = output_shape
        self.activation = activation
        self.model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_shape[0]*output_shape[1],
            activation=activation
        )

    def forward(self, y):
        B_vec = self.model(y)
        B_matrix = B_vec.view(-1, self.output_shape[0], self.output_shape[1])
        return  B_matrix

# --- Full model including encoder, space varying Koopman operator, and decoder, and bilinear term for input 
#     based on control affine model on Eurika Kaiser 2021 paper ---#
class CombinedKoopman_withAux_withInput(nn.Module):
    def __init__(self, encoder, states_auxiliary, koopman, decoder, inputs_auxiliary):
        super().__init__()
        self.encoder = encoder
        self.states_auxiliary = states_auxiliary
        self.koopman = koopman
        self.decoder = decoder
        self.inputs_auxiliary = inputs_auxiliary

    def forward(self, x):
        y = self.encoder(x)
        Lambda = self.states_auxiliary(y)
        y_plus = self.koopman(y, Lambda)
        x_plus = self.decoder(y_plus)
        B_vec = self.inputs_auxiliary(y)

        # Placeholder for the gradient
        grad_encoder = torch.zeros((y.shape[1], x.shape[1]), dtype=torch.float32)
        for i in range(y.shape[1]):
            if x.grad is not None: 
                x.grad.zero_()                         # Zero out previous gradients
            grad_output = torch.zeros_like(y)          # Create a zeros tensor with the same shape as y0
            grad_output[:, i] = 1                       # Set the i-th element of grad_output to 1
            y.backward(gradient=grad_output, retain_graph=True)    # Compute gradients for the i-th output with respect to x0
            grad_encoder[i] = x.grad.detach()              # Copy the computed gradients (for the i-th output) into the Jacobian
        # print("Jacobian:", grad)

        return y, y_plus, x_plus, grad_encoder, B_vec
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        for i in range(len(self.states_auxiliary.net)):
            self.states_auxiliary.net[i].requires_grad_(status)
        self.decoder.requires_grad_(status)
        self.inputs_auxiliary.requires_grad_(status)

    def Kmat(self, y, omega):
        w = omega[:,0]
        mu = w = omega[:,1]
        dt = 2E-2
        # Compute exp_term and rotation terms for each element in the batch
        exp_term = torch.exp(mu * dt).view(-1, 1, 1)
        cos_wdt = torch.cos(w * dt).view(-1, 1, 1)
        sin_wdt = torch.sin(w * dt).view(-1, 1, 1)

        # Create a batch of 2x2 rotation matrices
        zero = torch.zeros_like(cos_wdt)
        rot_term = torch.cat([torch.cat([cos_wdt, -sin_wdt], dim=2),
                              torch.cat([sin_wdt, cos_wdt], dim=2)], dim=1)
        
        B = exp_term * rot_term

        # Reshape y for batch matrix multiplication
        y_reshaped = y.view(y.shape[0], 2, -1) # Assuming y has shape [batch_size, 2 * n]

        y_plus = torch.matmul(B, y_reshaped)

        # Reshape y_plus back to original shape if necessary
        y_plus = y_plus.view(y.shape)
        return y_plus


# Obsolete models

# --- Linear Koopman Operator with Input---#
class LinearKoopmanWithInput(nn.Module):
    def __init__(
        self,
        lifted_states_dimension,
        inpus_dimension,
    ):
        super().__init__()
        self.lifted_states_dimension = lifted_states_dimension
        self.inpus_dimension = inpus_dimension
        self.A = nn.Linear(lifted_states_dimension, lifted_states_dimension, bias=False)
        self.B = nn.Linear(inpus_dimension, lifted_states_dimension, bias=False)

    def forward(self, x, u):
        y = self.A(x) + self.B(u)
        return y, y