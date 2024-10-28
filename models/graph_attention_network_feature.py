import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class  graph_convolution(nn.Module):
    def __init__(self, in_features, out_features, node_n=21, seq_len=40, num_heads=1, bias=True):
        super( graph_convolution, self).__init__()

       
        self.feature_weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
       
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.num_heads = num_heads

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.feature_weights.size(1))
        self.feature_weights.data.uniform_(-stdv, stdv)
       
       
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        B,K,F = v.shape
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        
        return combined.view(v.size(0), K, K, 2 * F)

    def forward(self, input):
        #print(input.shape)
        batch_size, features,num_nodes, seq_len = input.size()
        y = input.permute(0,3,2,1)
        
        
        #y = torch.matmul(input, self.temporal_graph_weights)
        y = torch.matmul(y, self.feature_weights)
        
        batch_size,seq_len , num_nodes, out_features = y.size()

        # Apply multi-head attention
        y = y.permute(0,3,2,1) # bs, seq_len, outfeature,n
      
        #print(y.shape)
     
        if self.bias is not None:
            return (y + self.bias)
        else:
            return y


            
class residual_graph_convolution(nn.Module):
    def __init__(self, features, node_n=21, seq_len = 40, bias=True, p_dropout=0.3,num_heads = 4):
        super(residual_graph_convolution, self).__init__()
        
        self.gcn = graph_convolution(features, features, node_n=node_n, seq_len=seq_len, num_heads=num_heads, bias=bias)        
        self.ln = nn.LayerNorm([features, node_n, seq_len], elementwise_affine=True)                      
        self.act_f = nn.Tanh()
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, x):
        
        y = self.gcn(x)
        y = self.ln(y)
        y = self.act_f(y)
        y = self.dropout(y)        
        return y + x
        
        
class graph_convolution_network(nn.Module):
    def __init__(self, in_features, latent_features, node_n=21, seq_len=40, p_dropout=0.3, residual_gcns_num=1):
        super(graph_convolution_network, self).__init__()
        self.residual_gcns_num = residual_gcns_num
        self.seq_len = seq_len
        
        self.start_gcn = graph_convolution(in_features=in_features, out_features=latent_features, node_n=node_n, seq_len=seq_len,num_heads=4)
        
        self.residual_gcns = []
        for i in range(residual_gcns_num):
            self.residual_gcns.append(residual_graph_convolution(features=latent_features, node_n=node_n, seq_len=seq_len*2, p_dropout=p_dropout,num_heads=4))        
        self.residual_gcns = nn.ModuleList(self.residual_gcns)
  
        self.end_gcn = graph_convolution(in_features=latent_features, out_features=in_features, node_n=node_n, seq_len=seq_len,num_heads=4)
        
    def forward(self, x):
        #print(x.shape)
        y = self.start_gcn(x)
        
        y = torch.cat((y, y), dim=3)
        for i in range(self.residual_gcns_num):
            y = self.residual_gcns[i](y)
        y = y[:, :, :, :self.seq_len]
        
        y = self.end_gcn(y)
        
        return y+x