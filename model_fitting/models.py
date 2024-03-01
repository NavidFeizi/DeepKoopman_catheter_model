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
        hidden_sizes,
        output_size,
        activation=None,
        normalizer_params=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.auto_normalize = False
        assert len(hidden_sizes) >= 1, "There must be at least one hidden layer"
        # input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        # hidden layers
        hidden = []
        i = -1
        for i in range(len(hidden_sizes) - 1):
            hidden.append((nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
        self.hidden_layers = torch.nn.ModuleList(hidden)
        # output layer
        self.output_layer = nn.Linear(hidden_sizes[i + 1], output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            # self.act = getattr(nn.activation, activation)()
            self.act = nn.ReLU()
        else:
            self.act = activation

        if normalizer_params is not None:
            self.normalizer_params = normalizer_params
            self.x_min = np.array(normalizer_params["params_x"]["min"])
            self.x_max = np.array(normalizer_params["params_x"]["max"])
            self.y_min = np.array(normalizer_params["params_y"]["min"])
            self.y_max = np.array(normalizer_params["params_y"]["max"])

    def forward(self, x):
        # normalize the input if necessary
        if self.auto_normalize:
            x = (x - self.x_min) / (self.x_max - self.x_min)
        # itterdate over layers
        x = self.input_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        y = self.output_layer(x)
        # de normalize the input if necessary
        if self.auto_normalize:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y

# --- Linear Koopman Operator ---#
class LinearKoopman(nn.Module):
    def __init__(
        self,
        dimension,
    ):
        super().__init__()
        self.dimension = dimension
        self.linear_model_layer = nn.Linear(dimension, dimension, bias=False)

    def forward(self, x):
        y = self.linear_model_layer(x)
        return y, y

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

# --- Full Model ---#
class CombinedDeepKoopman(nn.Module):
    def __init__(self, encoder, koopman, decoder):
        super().__init__()
        self.encoder = encoder
        self.koopman = koopman
        self.decoder = decoder

    def forward(self, x):
        y = self.encoder(x)
        y_plus, _ = self.koopman(y)
        x_plus = self.decoder(y_plus)
        return y, y_plus, x_plus
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        self.koopman.requires_grad_(status)
        self.decoder.requires_grad_(status)

# --- Full Model ---#
class Auxilary(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()
        self.input_size = input_size 
        self.params = params
        self.net = nn.ModuleList()
        for i in range(params["num_complexeigens_pairs"]):
            self.net.append(MLP(input_size=1, hidden_sizes=params["hidden_sizes"], output_size=2, activation=params["activation"]))

        for i in range(params["num_realeigens"]):
            self.net.append(MLP(input_size=1, hidden_sizes=params["hidden_sizes"], output_size=1, activation=params["activation"]))

    def forward(self, y):
        Lambdas = []
        for i in range(self.params["num_complexeigens_pairs"]):
            yblock = y[:, i*2:i*2+2]
            yradius = torch.sqrt(torch.mean(yblock**2, dim=1)).unsqueeze(dim=1)
            Lambdas.append(self.net[i](yradius))

        for i in range(self.params["num_realeigens"]):
            yblock = y[:, self.params["num_complexeigens_pairs"]*2 + i].unsqueeze(dim=1) 
            Lambdas.append(self.net[self.params["num_complexeigens_pairs"]+i](yblock))
        Lambdas = torch.cat(Lambdas, dim=1)  # Use torch.cat for proper tensor concatenation
        return  Lambdas

# --- Universal Varying koopman Operator based on Bethany Lusch Nature paper ---#
class TrainableKoopmanDynamics(nn.Module):
    def __init__(self, n, m, dt):
        """
        Initialize the module with n complex Jordan blocks and m real eigenvalues.
        Dimension d = 2*n + m, since each complex Jordan block is 2-dimensional.

        Args:
        n (int): Number of complex Jordan blocks.
        m (int): Number of real eigenvalues.
        dt (float): Time step delta t.
        """
        super(TrainableKoopmanDynamics, self).__init__()
        self.n = n  # num Complex Pairs of Eigen values
        self.m = m  # num Real Eigen values
        self.dt = dt
        self.dimension = 2*n + m

        self.mu = None
        self.omega = None
        self.real_eigenvalues = None

    def forward(self, x, Lambda):
        batch_size = x.shape[0]

        # Update mu, omega, and real_eigenvalues based on Lambda
        self.mu = Lambda[:, 0:self.n]  # Adjusted slicing for batch handling
        self.omega = Lambda[:, self.n:2*self.n]
        self.real_eigenvalues = Lambda[:, 2*self.n:]

        # Initialize A as a batch of identity matrices
        A = torch.eye(self.dimension, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Update A for complex eigenvalues
        for i in range(self.n):
            exp_mu_dt = torch.exp(self.mu[:, i] * self.dt)
            cos_omega_dt = torch.cos(self.omega[:, i] * self.dt)
            sin_omega_dt = torch.sin(self.omega[:, i] * self.dt)

            # Constructing the 2x2 block for each item in the batch
            block = torch.stack([
                exp_mu_dt * cos_omega_dt, -exp_mu_dt * sin_omega_dt,
                exp_mu_dt * sin_omega_dt, exp_mu_dt * cos_omega_dt
            ], dim=-1).view(batch_size, 2, 2)

            # Placing the block in the correct position for each A in the batch
            A[:, 2*i:2*(i+1), 2*i:2*(i+1)] = block

        # Update A for real eigenvalues
        for i in range(self.m):
            exp_lambda_dt = torch.exp(self.real_eigenvalues[:, i] * self.dt)
            A[:, 2*self.n + i, 2*self.n + i] = exp_lambda_dt

        # Compute y = A * x for each item in the batch
        x = x.unsqueeze(-1)  # Add an extra dimension for matmul
        y = torch.matmul(A, x).squeeze(-1)  # Remove the extra dimension after matmul

        return y

# --- Full Model ---#
class CombinedKoopman_withAux(nn.Module):
    def __init__(self, encoder, auxilary, koopman, decoder):
        super().__init__()
        self.encoder = encoder
        self.auxilary = auxilary
        self.koopman = koopman
        self.decoder = decoder

    def forward(self, x):
        y = self.encoder(x)
        Lambda = self.auxilary(y)
        y_plus = self.koopman(y, Lambda)
        x_plus = self.decoder(y_plus)
        return y, y_plus, x_plus
    
    def requires_grad_(self, status):
        self.encoder.requires_grad_(status)
        for i in range(len(self.auxilary.net)):
            self.auxilary.net[i].requires_grad_(status)
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






















# # --- Temporal Convolutional Network combined with MLP ---#
# class TCN(nn.Module):
#     def __init__(self, input_size, hidden_sizes_TCN, hidden_sizes_MLP):
#         super().__init__()

#         layers = []
#         kernel_size = 4

#         in_channels = input_size
#         for h in hidden_sizes_TCN:
#             layers.append(
#                 nn.Conv2d(
#                     in_channels,
#                     hidden_sizes_TCN,
#                     kernel_size=kernel_size,
#                     padding=(kernel_size - 1) // 2,
#                 )
#             )
#             layers.append(nn.ReLU())
#             in_channels = h

#         for h in hidden_sizes_MLP - 1:
#             layers.append(nn.Linear(hidden_sizes_MLP, hidden_sizes_MLP[0]))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_sizes_MLP, 1))
#         self.tcn = nn.Sequential(*layers)

#     def forward(self, x):
#         x, _ = self.tcn(x)
#         # x = self.linear(x[:, -1, :]  )      # do not return sequnecy to dense layer
#         return x


# ''' Temporal Convolutional Network '''
# class TemporalConvolutionalNetwork(mc.TorchMLCasadiModule):
#     def __init__(
#         self, input_size, hidden_sizes, kernel_size=3, seq_length=10, activation=None
#     ):
#         super().__init__()

#         layers = []
#         in_channels = input_size
#         layers.append(
#             mc.nn.Conv1d_4(
#                 in_channels=in_channels,
#                 out_channels=hidden_sizes[0],
#                 kernel_size=kernel_size,
#                 padding=int((kernel_size - 1) / 2),
#             )
#         )
#         in_channels = hidden_sizes[0]
#         for h in hidden_sizes[1:]:
#             layers.append(
#                 mc.nn.Conv1d_8(
#                     in_channels=in_channels,
#                     out_channels=h,
#                     kernel_size=kernel_size,
#                     padding=int((kernel_size - 1) / 2),
#                 )
#             )
#             in_channels = h
#         self.tcn_layers = torch.nn.ModuleList(layers)
#         # layers.append(self.act)

#         self.output_layer = mc.nn.Conv1d_out(
#             in_channels=in_channels,
#             out_channels=h,
#             kernel_size=seq_length,
#             padding=0,
#         )
#         # self.output_layer = mc.nn.AvgPool1d(kernel_size=seq_length, stride=1)
#         # self.tcn_layers = nn.Sequential(*layers)

#         if activation is None:
#             self.act = lambda x: x
#         elif type(activation) is str:
#             self.act = getattr(mc.nn.activation, activation)()
#         else:
#             self.act = activation

#     def forward(self, x):
#         for layer in self.tcn_layers:
#             x = layer(x)
#             x = self.act(x)
#         y = self.output_layer(x)
#         if len(y.shape) == 3:
#             return y[:, :, 0]
#         else:
#             return y[:, 0]


#############################
#     ML Casadi models     #
#############################


# # --- Multilayer Perceptron ---#
# class MLP_mc(mc.TorchMLCasadiModule):
#     def __init__(
#         self,
#         input_size,
#         state_size,
#         hidden_sizes,
#         output_size,
#         activation=None,
#         normalizer_params=None,
#     ):
#         super().__init__()
#         self.input_size = input_size
#         self.state_size = state_size
#         self.hidden_sizes = hidden_sizes
#         self.output_size = output_size

#         self.auto_normalize = False
#         assert len(hidden_sizes) >= 1, "There must be at least one hidden layer"
#         # input layer
#         self.input_layer = mc.nn.Linear(state_size + input_size, hidden_sizes[0])
#         # hidden layers
#         hidden = []
#         for i in range(len(hidden_sizes) - 1):
#             hidden.append((mc.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
#         self.hidden_layers = torch.nn.ModuleList(hidden)
#         # output layer
#         self.output_layer = mc.nn.Linear(hidden_sizes[i + 1], output_size)

#         if activation is None:
#             self.act = lambda x: x
#         elif type(activation) is str:
#             self.act = getattr(mc.nn.activation, activation)()
#         else:
#             self.act = activation

#         # self.mlp = nn.Sequential(
#         #     mc.nn.Linear(input_size, mlp_hidden_sizes[0]),
#         #     self.act,
#         # )
#         # for i in range(len(mlp_hidden_sizes) - 1):
#         #     self.mlp.add_module(
#         #         f"fc{i + 2}", mc.nn.Linear(mlp_hidden_sizes[i], mlp_hidden_sizes[i + 1])
#         #     )
#         #     self.mlp.add_module(f"relu{i + 2}", self.act)
#         #     # self.mlp.add_module(f"dropout{i + 2}", nn.Dropout(dropout))
#         # self.output_layer = mc.nn.Linear(mlp_hidden_sizes[-1], output_size)

#         if normalizer_params is not None:
#             self.normalizer_params = normalizer_params
#             self.x_min = np.array(normalizer_params["params_x"]["min"])
#             self.x_max = np.array(normalizer_params["params_x"]["max"])
#             self.y_min = np.array(normalizer_params["params_y"]["min"])
#             self.y_max = np.array(normalizer_params["params_y"]["max"])

#     def forward(self, x):
#         # normalize the input if necessary
#         if self.auto_normalize:
#             x = (x - self.x_min) / (self.x_max - self.x_min)
#         # itterdate over layers
#         x = self.input_layer(x)
#         x = self.act(x)
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self.act(x)
#         y = self.output_layer(x)
#         # de normalize the input if necessary
#         if self.auto_normalize:
#             y = y * (self.y_max - self.y_min) + self.y_min
#         return y


# # --- Multilayer Perceptron ---#
# class MLP_old(mc.TorchMLCasadiModule):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         output_size,
#         n_hidden,
#         activation=None,
#         normalizer_params=None,
#     ):
#         super().__init__()
#         self.auto_normalize = False
#         assert n_hidden >= 1, "There must be at least one hidden layer"
#         self.input_size = input_size
#         self.output_size = output_size
#         self.input_layer = mc.nn.Linear(input_size, hidden_size)

#         hidden = []
#         for i in range(n_hidden - 1):
#             hidden.append((mc.nn.Linear(hidden_size, hidden_size)))
#         self.hidden_layers = torch.nn.ModuleList(hidden)

#         self.output_layer = mc.nn.Linear(hidden_size, output_size)

#         if activation is None:
#             self.act = lambda x: x
#         elif type(activation) is str:
#             self.act = getattr(mc.nn.activation, activation)()
#         else:
#             self.act = activation

#         if normalizer_params is not None:
#             self.normalizer_params = normalizer_params
#             self.x_min = np.array(normalizer_params["params_x"]["min"])
#             self.x_max = np.array(normalizer_params["params_x"]["max"])
#             self.y_min = np.array(normalizer_params["params_y"]["min"])
#             self.y_max = np.array(normalizer_params["params_y"]["max"])

#     def forward(self, x):
#         if self.auto_normalize:
#             x = (x - self.x_min) / (self.x_max - self.x_min)
#         x = self.input_layer(x)
#         x = self.act(x)
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self.act(x)
#         y = self.output_layer(x)
#         if self.auto_normalize:
#             y = y * (self.y_max - self.y_min) + self.y_min
#         return y


# # --- Temporal Convolutional Network ---#
# class TCN_mc(mc.TorchMLCasadiModule):
#     def __init__(
#         self,
#         input_size,
#         hidden_sizes,
#         kernel_size=3,
#         seq_length=10,
#         activation=None,
#         normalizer_params=None,
#     ):
#         super().__init__()
#         self.auto_normalize = False
#         layers = []
#         in_channels = input_size
#         for h in hidden_sizes:
#             layers.append(
#                 mc.nn.Conv1d(
#                     in_channels=in_channels,
#                     out_channels=h,
#                     kernel_size=kernel_size,
#                     padding=int((kernel_size - 1) / 2),
#                 )
#             )
#             in_channels = h
#         self.tcn_layers = torch.nn.ModuleList(layers)
#         # layers.append(self.act)

#         self.output_layer = mc.nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=h,
#             kernel_size=seq_length,
#             padding=0,
#         )
#         # self.output_layer = mc.nn.AvgPool1d(kernel_size=seq_length, stride=1)
#         # self.tcn_layers = nn.Sequential(*layers)

#         if activation is None:
#             self.act = lambda x: x
#         elif type(activation) is str:
#             self.act = getattr(mc.nn.activation, activation)()
#         else:
#             self.act = activation

#         if normalizer_params is not None:
#             self.normalizer_params = normalizer_params
#             self.x_min = np.array(normalizer_params["params_x"]["min"])
#             self.x_max = np.array(normalizer_params["params_x"]["max"])
#             self.y_min = np.array(normalizer_params["params_y"]["min"])
#             self.y_max = np.array(normalizer_params["params_y"]["max"])

#     def forward(self, x):
#         # normalize the input if necessary
#         if self.auto_normalize:
#             x = (x - self.x_min) / (self.x_max - self.x_min)
#         for layer in self.tcn_layers:
#             x = layer(x)
#             x = self.act(x)
#         y = self.output_layer(x)
#         if len(y.shape) == 3:
#             y = y[:, :, 0]
#         else:
#             y = y[:, 0]
#         # de normalize the input if necessary
#         if self.auto_normalize:
#             y = y * (self.y_max - self.y_min) + self.y_min
#         return y


# # --- Temporal Convolutional Network combined with MLP ---#
# class TCNMLP_mc(mc.TorchMLCasadiModule):
#     def __init__(
#         self,
#         input_size,
#         state_size,
#         tcn_hidden_sizes,
#         mlp_hidden_sizes,
#         output_size,
#         tcn_kernel_size=3,
#         activation=None,
#         seq_length=10,
#         normalizer_params=None,
#     ):
#         super().__init__()
#         self.auto_normalize = False
#         # super(TCNMLP, self).__init__()
#         self.input_size = input_size
#         self.state_size = state_size
#         self.tcn_hidden_sizes = tcn_hidden_sizes
#         self.mlp_hidden_sizes = mlp_hidden_sizes
#         self.output_size = output_size
#         self.sequence_len = seq_length

#         if activation is None:
#             self.act = lambda x: x
#         elif type(activation) is str:
#             self.act = getattr(mc.nn.activation, activation)()
#         else:
#             self.act = activation

#         self.tcn = TCN_mc(
#             input_size=state_size + input_size,
#             hidden_sizes=tcn_hidden_sizes,
#             kernel_size=tcn_kernel_size,
#             activation=activation,
#             seq_length=seq_length,
#             normalizer_params=None,
#         )
#         self.mlp = MLP_mc(
#             state_size=tcn_hidden_sizes[-1],
#             input_size=0,
#             hidden_sizes=mlp_hidden_sizes,
#             output_size=output_size,
#             activation=None,
#             normalizer_params=None,
#         )

#         if normalizer_params is not None:
#             self.normalizer_params = normalizer_params
#             self.x_min = np.array(normalizer_params["params_x"]["min"])
#             self.x_max = np.array(normalizer_params["params_x"]["max"])
#             self.y_min = np.array(normalizer_params["params_y"]["min"])
#             self.y_max = np.array(normalizer_params["params_y"]["max"])

#     def forward(self, x):
#         if self.auto_normalize:
#             x = (x - self.x_min) / (self.x_max - self.x_min)
#         tcn_output = self.tcn(x)
#         # tcn_output = tcn_output.mean(-1 )  # Global average pooling over the temporal dimension
#         y = self.mlp(tcn_output)
#         if self.auto_normalize:
#             y = y * (self.y_max - self.y_min) + self.y_min
#         return y
