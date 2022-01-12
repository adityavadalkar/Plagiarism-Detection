import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
from src.gmlp import gMLP

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        super(SentAttNet, self).__init__()
        self.vector_size = sent_hidden_size
        self.sent_weight = nn.Parameter(torch.Tensor(sent_hidden_size, sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(sent_hidden_size, 1))
        self.gmlp = gMLP(num_tokens=0, dim=self.vector_size, seq_len=330, ff_mult=2, heads=1, attn_dim=32, depth=1, circulant_matrix = False)
        # self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        # self._create_weights(mean=0.0, std=0.05)

    # def _create_weights(self, mean=0.0, std=0.05):
    #     self.sent_bias.data.normal_(mean,std)
    #     self.sent_weight.data.normal_(mean, std)
    #     self.context_weight.data.normal_(mean, std)

    def forward(self, input):

        # f_output, h_output = self.gru(input, hidden_state)
        output = self.gmlp(input)
        #print('------------------>')
        #print(f_output.shape)
        # output = matrix_mul(output, self.sent_weight, self.sent_bias)
        # print("sent gmlp output ",output.shape)
        #print()
        # temp_output = matrix_mul(output, self.context_weight).permute(1,0)
        # temp_output = F.softmax(temp_output,dim=1)
        # print("sent temp output ", temp_output.shape)
        # output = element_wise_mul(output,temp_output.permute(1,0))
        # print("sent output ", output.shape)
        #output = self.fc(output)
        
        return output


if __name__ == "__main__":
    abc = SentAttNet()
