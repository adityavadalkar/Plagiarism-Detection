import torch
import torch.nn as nn
from src.sent_att_model_bert import SentAttNet
import torch.nn.functional as F
from src.gmlp import gMLP

class HierGraphAttNet(nn.Module):
    def __init__(self, vector_size, sent_hidden_size=50, batch_size=256):
        super(HierGraphAttNet, self).__init__()
        self.batch_size = batch_size
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(4*sent_hidden_size, sent_hidden_size)
        self.mlp_graph = nn.Linear(4* sent_hidden_size, 2*sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.r = nn.ReLU()
        self.sent_hidden_size = sent_hidden_size
        self.sent_gru = nn.GRU(vector_size, sent_hidden_size, bidirectional=True)
        self.sent_att_net = SentAttNet(sent_hidden_size, sent_hidden_size)
        self.new_sent_han = GMLP(num_tokens=0, dim=sent_hidden_size, seq_len=10, depth=6)
        # print('vector size: ', vector_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def encode(self, input):
        output = input.permute(1,0,2)
        #print(output.shape)
        output_list, hidden = self.sent_gru(output, self.sent_hidden_state)
        #print(output.shape)
        output_, output, hidden = self.new_sent_han(output_list)

        output = torch.cat((output_list,output.unsqueeze(0)),0)
        #output = torch.cat(output_list, 0)
        return output.permute(1,0,2)


    def graph_match(self, input_1, input_2):
        #Calculates cross document attention between the document and summary vectors
        
        attention_x = self.gmlp(input_1)
        attention_y = self.gmlp(input_2)
        print("att x ", attention_x.shape)
        output_x = torch.cat((input_1, attention_y), dim=2)
        output_y = torch.cat((input_2, attention_x), dim=2)

        output_x = self.mlp_graph(output_x)
        output_y = self.mlp_graph(output_y)
        return output_x, output_y

    def forward(self, input_1, input_2):
        output_1 = self.encode(input_1)

        output_2 = self.encode(input_2)
        print("output 2", output_2.shape)
        output_1, output_2 = self.graph_match(output_1, output_2) 
        
        if self.training:
            output_1 = output_1[:,-1,:].squeeze() # [128,6,100] -> [128,1,100]
            output_2 = output_2[:,-1,:].squeeze() 
            output = torch.cat((output_1, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))

            return self.m(output)
        else:
            output_1 = output_1.permute(1,0,2)
            output_2 = output_2[:,-1,:].squeeze()
            output_2 = torch.unsqueeze(output_2, 0)
            output_2 = output_2.expand(output_1.size())
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)

