import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import init_weights, get_padding
import math
from .mamba.mamba_ssm.modules.mamba_simple import Mamba
from .ESN_py.utils.sparseLIB import sparseESN
from .ESN_py.utils.func import check_dim, check_type, metric_func
from .ESN_py.learning_algorithm.Gradient_descent import gd_init, gd_rule, batch_gd_rule
LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1,3,5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1, 
                       dilation=dilation[0],                               
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[1],                               
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[2],                               
                       padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
            
class Generator_ESN(sparseESN):
    def __init__(self, args):
        super(Generator_ESN, self).__init__(args)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        
    def forward(self, input_source):
        
        batch_size, n_input, input_feature = input_source.shape
        Y = torch.zeros(batch_size, 172, 80, dtype=torch.float, device=self.device, requires_grad=False) # prediction data
        self.batch_x = torch.zeros((batch_size, self.resSize), device=self.device)


        for t in range(n_input):
            
            # Making input tensor
            u = input_source[:,t,:].reshape(batch_size, -1)
            # Update reservoir state
            self.batch_x = self.batch_update_state(u)
            extended_state = torch.hstack([self.ones, u, self.batch_x])
            # Forward pass
            Y[:,t,:] = torch.matmul(extended_state, self.Wout)
        return Y.permute(0,2,1)

class Generator_Mamba(nn.Module):
    def __init__(self, args):
        super(Generator_Mamba, self).__init__()
        USE_CUDA = torch.cuda.is_available()
        
        self.input_dim = args.input_dim
        self.d_model = args.d_model
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.n_layers = args.n_layers
        self.output_dim = args.output_dim
        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        
        # Initial projection layer to project input to d_model
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Layer Normalization as the first layer
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Stack n Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            ) 
            for _ in range(self.n_layers)
        ])
        
        # Final Classification Layer
        self.fc = nn.Linear(self.d_model, self.output_dim)
        
    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)

        # Residual connection between input_projection and Mamba layers
        
        #residual = x

        # Pass through stacked Mamba layers with residual connections
        for layer in self.mamba_layers:
            x = self.layer_norm(x)
            x = layer(x) #+ residual  # Residual connection at each layer
            #residual = x  # Update residual for the next layer
        
        # Final residual connection (input + final output)
        output = self.fc(x)  # Combine global residual
        return output.permute(0,2,1)



class Discriminator(torch.nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.h = h
        self.ch_init_downsample = h.ch_init_downsample
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.n_classes = h.n_classes
        self.input_size = h.input_size
        self.m = 1
        
        for j in range(len(h.downsample_rates)):
            self.m = self.m * h.downsample_rates[j]
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_downsample,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, 
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample*(2**i), 
                       h.ch_init_downsample*(2**(i+1)),
                       k, u, padding=math.ceil((k-u)/2))))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample*(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        
        self.GRU = nn.GRU(ch, ch//2,
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9,1)))
        
        # FC Layer 
        self.adv_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), 1),
            nn.Sigmoid())
        self.aux_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), h.n_classes),
            nn.Softmax(dim=1))
        
        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        # FC Layer
        x = x.view(-1,
                   self.ch_init_downsample
                   *2*8*(self.input_size//self.m))
        validity = self.adv_classifier(x)
        label = self.aux_classifier(x)
        
        return validity, label

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
            