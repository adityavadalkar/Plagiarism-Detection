import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.dg_word_att_model import WordAttNet
import torch.nn.functional as F

class DHierGraphNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, tune, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(DHierGraphNet, self).__init__()
        self.batch_size = batch_size
        # self.cos = nn.CosineSimilarity()
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(2 * sent_hidden_size, sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.mlp_graph = nn.Linear(2* sent_hidden_size, 1*sent_hidden_size)
        self.r = nn.ReLU()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, tune, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size

    def encode(self, input_1, input_2):
        word_outputs_list = []
        for i in input_1:
            word_outputs= self.word_att_net(i.permute(1, 0))
            word_outputs_list.append(word_outputs)

        word_outputs_1 = torch.cat(word_outputs_list, 0)
        # For the second passage:
        self._init_hidden_state(input_2.size(1))
        word_outputs_list = []
        for i in input_2:
            word_outputs = self.word_att_net(i.permute(1, 0))
            word_outputs_list.append(word_outputs)
        word_outputs_2 = torch.cat(word_outputs_list, 0)

        # one direction graph matching
        # att_output_1 = torch.cat((word_outputs_2,output_2),0)
        # att_output_2 = torch.cat((word_outputs_1, output_1),0)
        
        sents_1 = self.graph_match_one_direction(word_outputs_2, word_outputs_1)
        sents_2 = self.graph_match_one_direction(word_outputs_1, word_outputs_2)
        # print(sents_1.shape)
        # print(sents_2.shape)
        # sents_1 = word_outputs_1
        # sents_2 = word_outputs_2
        # print(word_outputs_1.shape)
        # print(word_outputs_2.shape)
        doc_1 = self.sent_att_net(sents_1)
        doc_2 = self.sent_att_net(sents_2)
        # print(doc_1.shape)
        # print(doc_2.shape)
        #bidirection graph mathching
        doc_eles_1 = torch.cat((sents_1,doc_1),0)
        doc_eles_2 = torch.cat((sents_2,doc_2),0)
        doc_eles_1 = doc_eles_1.permute(1, 0, 2)
        doc_eles_2 = doc_eles_2.permute(1, 0, 2)

        doc_1, doc_2 = self.graph_match(doc_eles_1, doc_eles_2)
        return doc_1, doc_2
    
    def graph_match(self, input_1, input_2):
        #print(input_1.shape)
        a = torch.matmul(input_1, input_2.permute(0,2,1)) # [128, 6, 100] \times [128, 100, 6] -> [128, 6, 6]
        a_x = F.softmax(a, dim=1)  # i->j [128, 6, 6]
        a_y = F.softmax(a, dim=0)  # j->i [128, 6, 6]
        #print(a_y.shape)
        #print(a_x.shape)
        attention_x = torch.matmul(a_x.transpose(1, 2), input_1)
        attention_y = torch.matmul(a_y, input_2) # [128, 6, 100]
        output_x = torch.cat((input_1, attention_y), dim=2)
        output_y = torch.cat((input_2, attention_x), dim=2)
        output_x = self.mlp_graph(output_x)
        output_y = self.mlp_graph(output_y)
        return output_x, output_y

    def graph_match_one_direction(self, input_1, input_2):
        input_1 = input_1.permute(1,0,2)
        input_2 = input_2.permute(1,0,2)
        # print("input 1 ",input_1.shape)
        # print("input 2 ",input_2.shape)
        a = torch.matmul(input_1, input_2.permute(0,2,1)) # [128, 6, 100] \times [128, 100, 6] -> [128, 6, 6]
        a_x = F.softmax(a, dim=1)  # i->j [128, 6, 6]
        a_y = F.softmax(a, dim=0)  # j->i [128, 6, 6]
        # print("ay ",a_y.shape)
        # print("ax ",a_x.shape)
        attention_x = torch.matmul(a_x.transpose(1, 2), input_1)
        # print("attx ",attention_x.shape)
        output_y = torch.cat((input_2, attention_x), dim=2)
        # print("output y ",output_y.shape)
        output_y = self.mlp_graph(output_y)
        output_y = output_y.permute(1,0,2)
        # print("mlp graph ",output_y.shape)
        return output_y


    def forward(self, input_1, input_2, in_test=False):
        #print(input.shape)
        input_1 = input_1.permute(1, 0, 2)
        input_2 = input_2.permute(1, 0, 2)
        output_1, output_2 = self.encode(input_1, input_2)
        if in_test:
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            output = self.m(output) #[128,6]
            return output
        else:
            output_1 = output_1[:,-1,:].squeeze() # [128,6,100] -> [128,1,100]
            output_2 = output_2[:,-1,:].squeeze() 
            output = torch.cat((output_1, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)