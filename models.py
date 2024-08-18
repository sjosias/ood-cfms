import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from torchcfm.models.unet import UNetModel
from typing import Tuple, Union, Callable
from torchdyn.models.cnf import CNF
from torchcfm.models.models import MLP

import copy
from torch.nn.utils.parametrizations import weight_norm
from torchvision import models
from torch.nn.functional import relu
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU()
}


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def sample_gaussian_like(y):
    return torch.randn_like(y)


def divergence_approx(f, y, e=None):
    # print("approximate", f.shape, y.shape, e)#, y.shape, e.shape)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    # print("approximate", f.shape, y.shape, e.shape, e_dzdx.shape)
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

   

def divergence_bf(f, z, **unused_kwargs):
    sum_diag = 0.
    f = f.reshape(f.shape[0], -1)
    z.requires_grad_(True)

    sum_diag = 0.
    # print("divergence brute force", f.requires_grad, z.requires_grad)
    for i in range(z.shape[1]):
        print(i, "divergence brute force", f[:, i].shape, z.shape)
        z.retain_grad()
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True, retain_graph=True)[0].contiguous()[:, i].contiguous()
        print(sum_diag)
        # break

    return sum_diag.contiguous()



class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # will be a Unet, trained from 
        self.diffeq = diffeq # differential equations - neural network will be a 
        self.residual = residual

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise. for huthinson
        if self._e is None:
            self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            
            dy = self.diffeq(t, y)
            
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                # print("using meth  brute force 1")
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                # print("using meth 2",  "approx")
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, -1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)

        # print("divergence", divergence.shape)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])

    

class ResizeUNetModel(UNetModel):
    """A UNetModel that does resizing before passing through the original Unet. Based off the UnetModel 
        in the torchCFM library, but reworked to resize for ODEFunc

    dim: tuple
        (batch_size?, num channels, W, H)

    num_channels: int, base channel count for the model.
    num_res_blocks: int, number of residual blocks per downsample, set to 1, not sure what this is yet
    num_heads: the number of attention heads in each attention layer
    num_heads_channels: if specified, ignore num_heads and instead use a fixed channel width per attention head.
    param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.

    """

    # TODO: Figure out what arguments to pass to the subclass
    def __init__(self, dim: Tuple[int], 
                 num_channels: int = 32, 
                 num_res_blocks: int = 2, 
                 image_channel: int = 1,
                 num_heads: int = 4,
                 num_head_channels: int = 64,
                 attention_resolutions: str = "16",
                 channel_mult = None,
                 fc = False,
                 dropout = 0.1):
        
        super().__init__(dim=dim, num_channels = num_channels, num_res_blocks=num_res_blocks, num_heads = num_heads, num_head_channels = num_head_channels, attention_resolutions= attention_resolutions, channel_mult= channel_mult, dropout=dropout)
     
        self.dim =  dim
        
        self.image_channels = image_channel
        


    def forward(self,t, x, **kwargs):
        
        dx = x
        dx = dx.view(-1, self.image_channels, int(self.dim[1]), int(self.dim[-1]))
        unet_output = super().forward(t, dx)
       
        unet_output = unet_output.view(-1, self.image_channels*self.dim[1]*self.dim[-1])
       
        return unet_output
    


class TorchDynCNFWrapper(CNF):
    def __init__(self, net:nn.Module, trace_estimator:Union[Callable, None]=None, noise_dist=None, order=1):
        """Continuous Normalizing Flow Wrapper class, since the original does not have provision for time

        :param net: function parametrizing the datasets vector field.
        :type net: nn.Module
        :param trace_estimator: specifies the strategy to otbain Jacobian traces. Options: (autograd_trace, hutch_trace)
        :type trace_estimator: Callable
        :param noise_dist: distribution of noise vectors sampled for stochastic trace estimators. Needs to have a `.sample` method.
        :type noise_dist: torch.distributions.Distribution
        :param order: specifies parameters of the Neural DE.
        :type order: int
        """
        super().__init__( net = net, trace_estimator=trace_estimator, noise_dist=noise_dist, order=order)
        

    def forward(self, t, x, **kwargs):
        with torch.set_grad_enabled(True):
            # first dimension is reserved to divergence propagation
            # x_in = x[:,1:].requires_grad_(True)
            print("cnf", x.requires_grad, t.requires_grad)
            x_in = x
            # print(t, x_in.shape, x.shape)
            # the neural network will handle the datasets-dynamics here
            if self.order > 1: x_out = self.higher_order(x_in) # TODO: incorporate time?
            else: x_out = self.net(t, x_in) # chang hhere

            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph



#####################################################################
################## image neural ode ################################# 
#####################################################################
    


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, 
        input_shape, 
        strides, 
        conv, 
        image_dim, 
        layer_type="concat", 
        nonlinearity="softplus", 
        num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        self.image_dim = image_dim
        if conv:
            # ConcatgConv2d is a custom functoin that calls nn.conv2d
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "concat": ConcatConv2d
            }[layer_type]
       

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        
        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y, **unused_kwargs):
        # print("inside diffeq", y.shape)
        dx = y.view(-1, 1, int(np.sqrt(self.image_dim)), int(np.sqrt(self.image_dim)))

        # print("inside diffeq", dx.shape, t.shape)
        # there is a squeeze layer in the cleanup branch of the cnf-robustness repo        
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)

        # print("before returning", dx.shape)
        dx = dx.view(-1, self.image_dim)
        return dx





class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()

        # added weight norm for gmm convergence testing
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        # module is conv2d
        self._layer = weight_norm(module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        ))

    def forward(self, t, x):
        # print('inside concatconv2d', x.shape, t.shape)
        t = t.reshape(-1, 1,1,1)

        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)        
        return self._layer(ttx)


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5, data_shape = (1,28,28)):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = None #{"max_iters": 300}
        self.data_shape = data_shape

        print("using solver {} with options {}".format(solver, self.solver_options))

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))
        # print(self.solver_options)
        # switch out 1 for number of channels
        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                options=self.solver_options,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]
        
        if len(self.data_shape) == 3:
            z_t = z_t.reshape(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]) 
            
        return z_t, logpz_t
    

    def num_evals(self):
        return self.odefunc._num_evals.item()

def sample_gaussian_like(y):
    return torch.randn_like(y)




################################################################################
############ UNET
#############################################################################
from unet_parts import DoubleConv, Down, Up, OutConv

class VanillaUNet(nn.Module):
    def __init__(self, dim, n_channels, n_classes, bilinear=False):
        super(VanillaUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dim = dim

        self.inc = (DoubleConv(n_channels, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.down3 = (Down(512, 1024))
        factor = 2 if bilinear else 1
        self.down4 = (Down(1024, 2048 // factor))
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

        # self.inc = (DoubleConv(n_channels, 64))
        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        # factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))

    def forward(self, t, x, **unused_kwargs):
        # unused kwargs are for the torchdyn interface
        # print("x shape", x.shape)
        x = x.view(-1, int(self.dim[0]), int(self.dim[1]), int(self.dim[-1]))
        
       
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print('inside unet', logits.shape)
        unet_output = logits.view(-1, self.dim[0]*self.dim[1]*self.dim[-1])
       
        return unet_output

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)





################################################################################
############ Fully connected network
#############################################################################
class FCNet(nn.Module):
    def __init__(self, in_out_dim: int = 2, hidden_dim: int = 32):
      super(FCNet, self).__init__()

      
      self.fc1 = nn.Linear(in_features = in_out_dim + 1 , out_features=hidden_dim, bias = True)
      self.fc2 = nn.Linear(in_features = hidden_dim , out_features=hidden_dim, bias = True)

      self.fc3 = nn.Linear(in_features = hidden_dim , out_features=hidden_dim, bias = True)
      self.fc4 = nn.Linear(in_features = hidden_dim , out_features=hidden_dim, bias = True)
      self.fc5 = nn.Linear(in_features = hidden_dim , out_features=hidden_dim, bias = True)
      self.fc6 = nn.Linear(in_features = hidden_dim , out_features=in_out_dim, bias = True)
      self.non_lin = NONLINEARITIES["softplus"]

    def forward(self, t, x):
        # print(x.shape, t)
        if len(t.shape) < 2:
            # print(t)
            t = t.expand((x.shape[0],1))
            # print("EXPAND")
        # print("inside FCNEt t.shape", t)
        x = torch.cat((x, t), dim=1)
        # print(x.shape)
        # print("after", x.shape, t.shape)
        x = self.non_lin(self.fc1(x))
        x = self.non_lin(self.fc2(x))
        x = self.non_lin(self.fc3(x))
        x = self.non_lin(self.fc4(x))
        x = self.non_lin(self.fc5(x))
        return self.fc6(x)



class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]
    



class ToyNet(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)
        self.t_buffer = []

    def reset_t_buffer(self):
        self.t_buffer = []


    def inner_forward(self, t, x):
        W, B, U = self.hyper_net(t)

        # print("inner forward", x.shap/e)
        Z = torch.unsqueeze(x, 0).repeat(self.width, 1, 1)
        # print("inner forward Z, W, B", Z.shape, W.shape, B.shape)
        h = torch.tanh(torch.matmul(Z, W) + B)
        # print("inner forward h, U", h.shape, U.shape)
        ret = torch.matmul(h, U).mean(0)
        # print("inner forward ret", ret.shape)
        return ret

    def forward(self, t, x):
        # self.t_buffer.append(t.item())
        z = x

        dz_dt = torch.zeros_like(x)
        with torch.set_grad_enabled(True):
            # z.requires_grad_(True)
            # print(t, len(t.shap/e))
            
            if len(t.shape) > 0:
                for idx, (tt, zz) in enumerate(zip(t,x)): 
                    # print("inside toynet", tt.shape, zz.shape)
    # 
                    dz_dtt = self.inner_forward(tt, zz)
                    dz_dt[idx, :] = dz_dtt
            else:
                
                dz_dt = self.inner_forward(t, x)
                # print("inside toynet", dz_dtt.shape, dz_dt.shape)
            # print ("dz_dt", dz_dt[:5])
            # Check dz_dt when done
            # dlogp_z_dt = -self.trace_df_dz(dz_dt, z).view(batchsize, 1)

        # print("x, dz_dt", x.shape, dz_dt.shape)
        return dz_dt


    def trace_df_dz(self, f, z):
        """Calculates the trace of the Jacobian df/dz.
        Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        """
        sum_diag = 0.
        for i in range(z.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

        return sum_diag.contiguous()




# Timestep embeddings froim https://wandb.ai/byyoung3/ml-news/reports/A-Gentle-Introduction-to-Diffusion---Vmlldzo2MzgxNjc3
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, time):
        
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    


class FCTimeEmbeddingNet(nn.Module):
    def __init__(self, data_dim: int = 1792):
        super().__init__()
        self.embedding_net = SinusoidalPositionEmbeddings(dim=data_dim)
        self.components = nn.ModuleList([nn.Linear(data_dim, data_dim) for i in range(3)])
        self.activation = nn.ReLU()
    
    def forward(self, t, x):
        if len(t.shape) == 0:
            t = torch.tensor([t.item()]).to(device)
            
        time_embedding = self.embedding_net(t)
        # print("time shape", time_embedding.shape)
        # print("x shape", x.shape)
        ret = torch.zeros_like(x)
        for idx, comp in enumerate(self.components):
            if idx == len(self.components) - 1:
                ret += comp(x)
            else:
                ret += self.activation(comp(x) + time_embedding)
        return ret
    

class MLPWrapper(MLP):
    def __init__(self, dim: int = 1792, out_dim=None, w=64, time_varying=False):
        super().__init__(dim, out_dim, w, time_varying)
        
        
    
    def forward(self, t, x):
        # print("t", t.shape)
        # print("x", x.shape)
        if len(t.shape) > 0:
            concat_xt = torch.cat([x, t.reshape(-1,1)], dim=1)
        else:
            t = t*torch.ones((x.shape[0], 1)).to(device)
            concat_xt = torch.cat([x, t], dim=1)

        # print("concat_xt", concat_xt.shape)

        return super().forward(concat_xt)




class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, t, x):
        # print(t.shape)
        if len(t.shape) > 0:
            # print(t.shape)
            concat_xt = torch.cat([x, t.reshape(-1,1)], dim=1)
        else:
            t = t*torch.ones((x.shape[0], 1)).to(device)
            concat_xt = torch.cat([x, t], dim=1)
        return self.net(concat_xt)
    




class SkipMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim


        self.down1 = torch.nn.Linear(dim + (1 if time_varying else 0), w)
        # self.down_ulayers = [torch.nn.Linear(w,w) for i in range(2)]
        self.down_u1 = torch.nn.Linear(w,w)
        self.down_u2 = torch.nn.Linear(w,w)
        self.bottleneck = torch.nn.Linear(w,w)
        self.up1 = torch.nn.Linear(w,w)
        self.up2 = torch.nn.Linear(2*w,w) 
        self.output_layer = torch.nn.Linear(2*w, out_dim)
        

    def forward(self, t, x):
        # print(t.shape)
        act = torch.nn.SiLU()
        # concat time in an appropriate manner
        if len(t.shape) > 0:
            # print(t.shape)
            concat_xt = torch.cat([x, t.reshape(-1,1)], dim=1)
        else:
            t = t*torch.ones((x.shape[0], 1)).to(device)
            concat_xt = torch.cat([x, t], dim=1)

        down_output = act(self.down1(concat_xt))
        # get activations for downlayers
        downu_act = [act(self.down_u1(down_output))]
        downu_act.append(act(self.down_u2(downu_act[0])))

        # Compute bottleneck layer
        bottleneck = act(self.bottleneck(downu_act[-1]))

        # compute up layers
        upsample_1 = act(self.up1(bottleneck))
        
        concat_upsample1 = torch.cat([upsample_1, downu_act[-1]], dim = 1)
        
        upsample_2 = act(self.up2(concat_upsample1))

        concat_upsample2 = torch.cat([upsample_2, downu_act[-2]], dim = 1)


        output = self.output_layer(concat_upsample2)
        return output
