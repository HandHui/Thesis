import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_net import ConvNet
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharEncoder(nn.Module):
    """
    Input:  (batch_size, seq_len, char_features)
    """
    def __init__(self, weight, channels, kernel_size, dropout, emb_dropout):  #weight 是word_embedding
        super(CharEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(weight, freeze=False)    ##加载词向量
        self.drop = nn.Dropout(emb_dropout)
        self.conv_net = ConvNet(channels, kernel_size, dropout, dilated=True, residual=False)
        #print(weight.size(0),weight.size(1))
    def forward(self,  char_input):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        #  -> (batch_size, seq_len, embedding_size + char_features)
        #  -> (batch_size, embedding_size + char_features, seq_len)
        embeddings=self.embed(char_input)
#         embeddings = torch.cat((embeddings, char_input), 2)                     ### 将字与词连接起来，以第2维度连接起来
        embeddings=embeddings.transpose(1, 2).contiguous()                      ### 矩阵再次转置

        #print("embeddings:----------",embeddings.size())

        # (batch_size, embedding_size + char_features, seq_len) -> (batch_size, conv_size, seq_len)
        conv_out = self.conv_net(self.drop(embeddings))

        # torch.cat(embeddings, conv_out), 1) ==>(batch_size, embedding_size + char_features, seq_len) -> (batch_size, conv_size + embedding_size + char_features, seq_len)
        #  -> (batch_size, seq_len, conv_size + embedding_size + char_features)
        return torch.cat((embeddings, conv_out), 1).transpose(1, 2).contiguous()



class LDecoder(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size,NUM_LAYERS):
        super(LDecoder, self).__init__()
        self.input_size=input_size
        self.hidden_dim = hidden_dim      ###有
        self.output_size=output_size      ### 分为多少类
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers = NUM_LAYERS)
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.init_weight()

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        #print(self.hidden_dim,self.input_size,self.output_size)
        # self.hidden = self.init_hidden(inputs.size(1))
        self.hidden = self.init_hidden(inputs.size(1))
        lstm_out, self.hidden = self.lstm(inputs,self.hidden)
        #lstm_out, self.hidden = self.lstm(inputs,None)
        y = self.hidden2label(lstm_out)
        return y

    def init_weight(self):
        nn.init.kaiming_uniform_(self.hidden2label.weight.data, mode='fan_in', nonlinearity='relu')

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).to(device),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).to(device))


class BLDecoder(nn.Module):  ###双向LSTM编码
    def __init__(self,input_size,hidden_dim,output_size,NUM_LAYERS):
        super(BLDecoder, self).__init__()
        self.input_size=input_size
        self.hidden_dim = hidden_dim      ###有
        self.output_size=output_size      ### 分为多少类
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers = NUM_LAYERS)
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.init_weight()

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        #print(self.hidden_dim,self.input_size,self.output_size)
        # self.hidden = self.init_hidden(inputs.size(1))
        self.hidden = self.init_hidden(inputs.size(1))
        lstm_out, self.hidden = self.lstm(inputs,self.hidden)
        #lstm_out, self.hidden = self.lstm(inputs,None)
        y = self.hidden2label(lstm_out)
        return y

    def init_weight(self):
        nn.init.kaiming_uniform_(self.hidden2label.weight.data, mode='fan_in', nonlinearity='relu')

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))               


class Model(nn.Module):
    def __init__(self,
                 weight, char_embedding_size, char_channels, char_kernel_size, num_tag, dropout, emb_dropout):    ##搭建网络结构
        super(Model, self).__init__()

        self.char_encoder = CharEncoder(weight, char_channels, char_kernel_size,dropout=dropout, emb_dropout=emb_dropout)
        self.drop = nn.Dropout(dropout)
#         self.char_conv_size = char_channels[-1]
        self.char_embedding_size = char_embedding_size
        self.char_conv_size = char_channels[-1]
        
        # self.decoder = LDecoder(self.char_embedding_size+self.char_conv_size,    ###输入特征
                               # 64 + self.char_embedding_size + self.char_conv_size,###隐藏层维度
                               # num_tag,NUM_LAYERS=1)   ###利用LSTM解码
							 
        self.decoder = BLDecoder(self.char_embedding_size+self.char_conv_size,    ###输入特征
                               64 + self.char_embedding_size + self.char_conv_size,###隐藏层维度
                               num_tag,NUM_LAYERS=1)      ### 利用双向LSTM解码

        self.init_weights()

    def forward(self, char_input):
        batch_size = char_input.size(0)  #32
        seq_len = char_input.size(1)    #句子长度
#         char_input=char_input.contiguous()
#         #print(char_input.view(-1, char_input.size(2)).size(0),char_input.view(-1, char_input.size(2)).size(1))##3200*10
#         char_output = self.char_encoder(char_input.view(-1, char_input.size(2))).view(batch_size, seq_len, -1)#char_input.size(2)==20  调用forward方法
        char_output = self.char_encoder(char_input)  ##(batch_size, seq_len, conv_size + embedding_size + char_features)
        y = self.decoder(char_output)

        return F.log_softmax(y, dim=2)
#         return F.log_softmax(y, dim=1)

    def init_weights(self):
        pass




#word_embeddings = torch.tensor(np.load("data/NYT_CoType/word2vec.vectors.npy"))
word_embeddings = torch.tensor(np.load("../data/char_embedding.npy"))
print(word_embeddings.shape)
dropout=(0.5,)
emb_dropout=0.25

if __name__ == "__main__":
    print('Hello world!!')
    # model=Model(charset_size=96, char_embedding_size=50, char_channels=[50, 50, 50, 50],
              # char_padding_idx=94, char_kernel_size=3, weight=word_embeddings,
              # word_embedding_size=300, word_channels=[350, 300, 300, 300],
              # word_kernel_size=3, num_tag=193, dropout=0.5,
              # emb_dropout=0.25)
    # print(model)
 # def attention_net(self,lstm_output,final_state):
        # lstm_output = lstm_output.permute(1,0,2)
        # hidden = final_state.squeeze(0)
        # attn_weights = torch.bmm(lstm_output,hidden.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = F.softmax(attn_weights,dim=2)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)         ##bmm  ----> batch matrix multiply
        # # ).squeeze(2)
        
        # return new_hidden_state