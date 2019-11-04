import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_net import ConvNet
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable


class CharEncoder(nn.Module):

    """
    Input: (batch_size, seq_len)
    Output: (batch_size, conv_size)
    """
    def __init__(self, char_num, embedding_size, channels, kernel_size, padding_idx, dropout, emb_dropout):    ###初始化时的参数
        super(CharEncoder, self).__init__()
        self.embed = nn.Embedding(char_num, embedding_size, padding_idx=padding_idx)                           ### ce=CharEncoder()时被调用  
        ### torch.nn.Embedding(m,n) m表示单词数目，n表示嵌入维度
        self.drop = nn.Dropout(emb_dropout)
        self.conv_net = ConvNet(channels, kernel_size, dropout=dropout)
        self.init_weights()
        #print(char_num,embedding_size)


    def forward(self, inputs):                                                                                ### 正儿八经调用时的参数
        seq_len = inputs.size(1)                                                                              ### pre = ce(inputs) 时调用
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size) -> (batch_size, embedding_size, seq_len)
        embeddings=self.embed(inputs)  #input:3200*10        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size) 
                                                             # 将seq_len长度的一句话换成seq_len*embedding_size
        embeddings = self.drop(embeddings)
        embeddings=embeddings.transpose(1, 2).contiguous()     ###转置矩阵，将第一维与第二维调换调换
                                                               ###  (batch_size, seq_len, embedding_size) -> (batch_size, embedding_size, seq_len)

        # (batch_size, embedding_size, seq_len) -> (batch_size, conv_size, seq_len)  词向量由行变列
        #  -> (batch_size, conv_size, 1) -> (batch_size, conv_size)
        return F.max_pool1d(self.conv_net(embeddings), seq_len).squeeze()       ###max_pool1d处理的是几个平面，将kernel_size个平面同时max_pool得到相应数量的切面
                                                                                ### 每个平面（在本例中每列）是一个词，取得了每一句话中，最有影响的一个词的相应数据
                                                                                ### 1 * embedding_size 
    def init_weights(self):
        nn.init.kaiming_uniform_(self.embed.weight.data, mode='fan_in', nonlinearity='relu')


class WordEncoder(nn.Module):
    """
    Input: (batch_size, seq_len), (batch_size, seq_len, char_features)
    """
    def __init__(self, weight, channels, kernel_size, dropout, emb_dropout):  #weight 是word_embedding
        super(WordEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(weight, freeze=False)    ##加载词向量
        self.drop = nn.Dropout(emb_dropout)
        self.conv_net = ConvNet(channels, kernel_size, dropout, dilated=True, residual=False)
        #print(weight.size(0),weight.size(1))
    def forward(self, word_input, char_input):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        #  -> (batch_size, seq_len, embedding_size + char_features)
        #  -> (batch_size, embedding_size + char_features, seq_len)
        embeddings=self.embed(word_input)
        embeddings = torch.cat((embeddings, char_input), 2)                     ### 将字与词连接起来，以第2维度连接起来
        embeddings=embeddings.transpose(1, 2).contiguous()                      ### 矩阵再次转置

        #print("embeddings:----------",embeddings.size())

        # (batch_size, embedding_size + char_features, seq_len) -> (batch_size, conv_size, seq_len)
        conv_out = self.conv_net(self.drop(embeddings))

        # torch.cat(embeddings, conv_out), 1) ==>(batch_size, embedding_size + char_features, seq_len) -> (batch_size, conv_size + embedding_size + char_features, seq_len)
        #  -> (batch_size, seq_len, conv_size + embedding_size + char_features)
        return torch.cat((embeddings, conv_out), 1).transpose(1, 2).contiguous()

#self.char_conv_size+self.word_embedding_size+self.word_conv_size, num_tag

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size,NUM_LAYERS):
        super(Decoder,self).__init__()
        self.input_size=input_size
        self.hidden_dim = hidden_dim      ###一个cell中有多少个神经元
        self.output_size=output_size      ### 分为多少类
        self.rnn = nn.RNN(input_size, hidden_dim, num_layers = NUM_LAYERS)  ##输入数据格式的特征数
        ###                in_feature,hidden_dim（隐藏层神经元个数） 自定义
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.init_weight()
        
        
   
        
        
        
    
    def forward(self,inputs):
    
        inputs = inputs.transpose(0,1)   ## (batch_size, seq_len, conv_size + embedding_size + char_features) -->(seq_len,batch_size,  conv_size + embedding_size + char_features)
        # print("inputs.shape:",inputs.size())   ## (64,195,650) (batch_size, seq_len, conv_size + embedding_size + char_features)
        # print("inputs.size()",inputs.size(0),inputs.size(1),inputs.size(2))
        # self.lstm.flatten_parameters()
        # print("")
        self.hidden = self.init_hidden(inputs.size(1))         ## 即seq_len,这个hidden作为输入，其值等于seq_len
                                                               ## 它与hidden_dim无关，hidden_dim是指一个cell内的，而它说的是有多少cell
                                                               ## 每一个hidden便可以包含hidden_dim个神经元
                                                               ## 即经过处理后，hidden.shape -->(1,batch_size,hidden_dim),为每一个batch提供一个（hidden_dim）维的初始权重
       
        output, self.hidden = self.rnn(inputs,self.hidden)
        # hiddenn , cell = self.hidden
        
        
        output = output.transpose(0,1)
        y = self.hidden2label(output)
        return y
        # return res_output
        
        
    def init_weight(self):
        nn.init.kaiming_uniform_(self.hidden2label.weight.data, mode='fan_in', nonlinearity='relu')

    def init_hidden(self, batch_size):                           ###很重要
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim))
                )

                

                

# class Decoder(nn.Module):
    # def __init__(self,input_size,hidden_dim,output_size,NUM_LAYERS):
        # super(Decoder, self).__init__()
        # self.input_size=input_size
        # self.hidden_dim = hidden_dim      ###有
        # self.output_size=output_size      ### 分为多少类
        # self.lstm = nn.LSTM(input_size, hidden_dim, num_layers = NUM_LAYERS)
        # self.hidden2label = nn.Linear(hidden_dim, output_size)
        # self.init_weight()

    # def forward(self, inputs):
        # self.lstm.flatten_parameters()
        # #print(self.hidden_dim,self.input_size,self.output_size)
        # # self.hidden = self.init_hidden(inputs.size(1))
        # self.hidden = self.init_hidden(inputs.size(1))
        # lstm_out, self.hidden = self.lstm(inputs,self.hidden)
        # #lstm_out, self.hidden = self.lstm(inputs,None)
        # y = self.hidden2label(lstm_out)
        # return y

    # def init_weight(self):
        # nn.init.kaiming_uniform_(self.hidden2label.weight.data, mode='fan_in', nonlinearity='relu')

    # def init_hidden(self, batch_size):
        # return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                # autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

class Model(nn.Module):
    def __init__(self, charset_size, char_embedding_size, char_channels, char_padding_idx, char_kernel_size,
                 weight, word_embedding_size, word_channels, word_kernel_size, num_tag, dropout, emb_dropout):    ##搭建网络结构
        super(Model, self).__init__()
        self.char_encoder = CharEncoder(charset_size, char_embedding_size, char_channels, char_kernel_size,
                                        char_padding_idx, dropout=dropout, emb_dropout=emb_dropout)                ### 调用CharEncoder的init方法
        self.word_encoder = WordEncoder(weight, word_channels, word_kernel_size,
                                        dropout=dropout, emb_dropout=emb_dropout)
        self.drop = nn.Dropout(dropout)
        self.char_conv_size = char_channels[-1]
        self.word_embedding_size = word_embedding_size
        self.word_conv_size = word_channels[-1]
        #self.decoder = nn.Linear(self.char_conv_size+self.word_embedding_size+self.word_conv_size, num_tag)
        self.decoder = Decoder(self.char_conv_size+self.word_embedding_size+self.word_conv_size,    ###输入特征
                               self.char_conv_size + self.word_embedding_size + self.word_conv_size,###隐藏层维度
                               num_tag,NUM_LAYERS=1)

        self.init_weights()

    def forward(self, word_input, char_input):
        batch_size = word_input.size(0)  #32
        seq_len = word_input.size(1)    #句子长度
        char_input=char_input.contiguous()
        #print(char_input.view(-1, char_input.size(2)).size(0),char_input.view(-1, char_input.size(2)).size(1))##3200*10
        char_output = self.char_encoder(char_input.view(-1, char_input.size(2))).view(batch_size, seq_len, -1)#char_input.size(2)==20  调用forward方法
        word_output = self.word_encoder(word_input, char_output)  ##(batch_size, seq_len, conv_size + embedding_size + char_features)
        y = self.decoder(word_output)

        return F.log_softmax(y, dim=2)
        # return F.log_softmax(y, dim=1)

    def init_weights(self):
        pass
        #self.decoder.bias.data.fill_(0)
        #nn.init.kaiming_uniform_(self.decoder.weight.data, mode='fan_in', nonlinearity='relu')

#word_embeddings = torch.tensor(np.load("data/NYT_CoType/word2vec.vectors.npy"))
word_embeddings = torch.tensor(np.load("../data/char_embedding.npy"))
print(word_embeddings.shape)
dropout=(0.5,)
emb_dropout=0.25

if __name__ == "__main__":
    model=Model(charset_size=96, char_embedding_size=50, char_channels=[50, 50, 50, 50],
              char_padding_idx=94, char_kernel_size=3, weight=word_embeddings,
              word_embedding_size=300, word_channels=[350, 300, 300, 300],
              word_kernel_size=3, num_tag=193, dropout=0.5,
              emb_dropout=0.25)
    print(model)
 # def attention_net(self,lstm_output,final_state):
        # lstm_output = lstm_output.permute(1,0,2)
        # hidden = final_state.squeeze(0)
        # attn_weights = torch.bmm(lstm_output,hidden.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = F.softmax(attn_weights,dim=2)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)         ##bmm  ----> batch matrix multiply
        # # ).squeeze(2)
        
        # return new_hidden_state