





# 机器翻译

机器翻译是指将一段文本从一种语言自动翻译到另一种语言。因为一段文本序列在不同语言中的长度不一定相同，所以我们使用机器翻译为例来介绍编码器—解码器和注意力机制的应用。

##  读取和预处理数据

我们先定义一些特殊符号。其中“&lt;pad&gt;”（padding）符号用来添加在较短序列后，直到每个序列等长，而“&lt;bos&gt;”和“&lt;eos&gt;”符号分别表示序列的开始和结束。



```python
!tar -xf d2lzh_pytorch.tar
```


```python
import collections
import os
import io
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data

import sys
# sys.path.append("..") 
import d2lzh_pytorch as d2l

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__, device)
```

    1.5.0 cpu


接着定义两个辅助函数对后面读取的数据进行预处理。


```python
# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens, all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens),
                        specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)
```

为了演示方便，我们在这里使用一个很小的法语—英语数据集。在这个数据集里，每一行是一对法语句子和它对应的英语句子，中间使用`'\t'`隔开。在读取数据时，我们在句末附上“&lt;eos&gt;”符号，并可能通过添加“&lt;pad&gt;”符号使每个序列的长度均为`max_seq_len`。我们为法语词和英语词分别创建词典。法语词的索引和英语词的索引相互独立。



```python
def read_data(max_seq_len):
    # in和out分别是input和output的缩写
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)
```

将序列的最大长度设成7，然后查看读取到的第一个样本。该样本分别包含法语词索引序列和英语词索引序列。


```python
max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
dataset[0]
```




    (tensor([ 5,  4, 45,  3,  2,  0,  0]), tensor([ 8,  4, 27,  3,  2,  0,  0]))



##  含注意力机制的编码器—解码器

我们将使用含注意力机制的编码器—解码器来将一段简短的法语翻译成英语。下面我们来介绍模型的实现。

###  编码器

在编码器中，我们将输入语言的词索引通过词嵌入层得到词的表征，然后输入到一个多层门控循环单元中。正如我们在6.5节（循环神经网络的简洁实现）中提到的，PyTorch的`nn.GRU`实例在前向计算后也会分别返回输出和最终时间步的多层隐藏状态。其中的输出指的是最后一层的隐藏层在各个时间步的隐藏状态，并不涉及输出层计算。注意力机制将这些输出作为键项和值项。


```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        embedding = self.embedding(inputs.long()).permute(1, 0, 2) # (seq_len, batch, input_size)
        return self.rnn(embedding, state)

    def begin_state(self):
        return None
```

下面我们来创建一个批量大小为4、时间步数为7的小批量序列输入。设门控循环单元的隐藏层个数为2，隐藏单元个数为16。编码器对该输入执行前向计算后返回的输出形状为(时间步数, 批量大小, 隐藏单元个数)。门控循环单元在最终时间步的多层隐藏状态的形状为(隐藏层个数, 批量大小, 隐藏单元个数)。对于门控循环单元来说，`state`就是一个元素，即隐藏状态；如果使用长短期记忆，`state`是一个元组，包含两个元素即隐藏状态和记忆细胞。


```python
encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
output, state = encoder(torch.zeros((4, 7)), encoder.begin_state())
output.shape, state.shape # GRU的state是h, 而LSTM的是一个元组(h, c)
```




    (torch.Size([7, 4, 16]), torch.Size([2, 4, 16]))



###  注意力机制

我们将实现10.11节（注意力机制）中定义的函数$a$：将输入连结后通过含单隐藏层的多层感知机变换。其中隐藏层的输入是解码器的隐藏状态与编码器在所有时间步上隐藏状态的一一连结，且使用tanh函数作为激活函数。输出层的输出个数为1。两个`Linear`实例均不使用偏差。其中函数$a$定义里向量$\boldsymbol{v}$的长度是一个超参数，即`attention_size`。


```python
def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
    return model
```

注意力机制的输入包括查询项、键项和值项。设编码器和解码器的隐藏单元个数相同。这里的查询项为解码器在上一时间步的隐藏状态，形状为(批量大小, 隐藏单元个数)；键项和值项均为编码器在所有时间步的隐藏状态，形状为(时间步数, 批量大小, 隐藏单元个数)。注意力机制返回当前时间步的背景变量，形状为(批量大小, 隐藏单元个数)。


```python
def attention_forward(model, enc_states, dec_state):
    """
    enc_states: (时间步数, 批量大小, 隐藏单元个数)
    dec_state: (批量大小, 隐藏单元个数)
    """
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=0)  # 返回背景变量
```

在下面的例子中，编码器的时间步数为10，批量大小为4，编码器和解码器的隐藏单元个数均为8。注意力机制返回一个小批量的背景向量，每个背景向量的长度等于编码器的隐藏单元个数。因此输出的形状为(4, 8)。


```python
seq_len, batch_size, num_hiddens = 10, 4, 8
model = attention_model(2*num_hiddens, 10) 
enc_states = torch.zeros((seq_len, batch_size, num_hiddens))
dec_state = torch.zeros((batch_size, num_hiddens))
attention_forward(model, enc_states, dec_state).shape
```




    torch.Size([4, 8])



###  含注意力机制的解码器

我们直接将编码器在最终时间步的隐藏状态作为解码器的初始隐藏状态。这要求编码器和解码器的循环神经网络使用相同的隐藏层个数和隐藏单元个数。

在解码器的前向计算中，我们先通过刚刚介绍的注意力机制计算得到当前时间步的背景向量。由于解码器的输入来自输出语言的词索引，我们将输入通过词嵌入层得到表征，然后和背景向量在特征维连结。我们将连结后的结果与上一时间步的隐藏状态通过门控循环单元计算出当前时间步的输出与隐藏状态。最后，我们将输出通过全连接层变换为有关各个输出词的预测，形状为(批量大小, 输出词典大小)。


```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2*num_hiddens, attention_size)
        # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 num_hiddens+embed_size
        self.rnn = nn.GRU(num_hiddens + embed_size, num_hiddens, 
                          num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[-1])
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1) 
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state
```

##  训练模型

我们先实现`batch_loss`函数计算一个小批量的损失。解码器在最初时间步的输入是特殊字符`BOS`。之后，解码器在某时间步的输入为样本输出序列在上一时间步的词，即强制教学。此外，同10.3节（word2vec的实现）中的实现一样，我们在这里也使用掩码变量避免填充项对损失函数计算的影响。


```python
def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是BOS
    dec_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)
    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失, 初始全1
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0
    l = torch.tensor([0.0])
    for y in Y.permute(1,0): # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
        mask = mask * (y != out_vocab.stoi[EOS]).float()
    return l / num_not_pad_tokens
```

在训练函数中，我们需要同时迭代编码器和解码器的模型参数。


```python
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
```

接下来，创建模型实例并设置超参数。然后，我们就可以训练模型了。


```python
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)
```

    epoch 10, loss 0.474
    epoch 20, loss 0.203
    epoch 30, loss 0.086
    epoch 40, loss 0.051
    epoch 50, loss 0.028


##  预测不定长的序列

在10.10节（束搜索）中我们介绍了3种方法来生成解码器在每个时间步的输出。这里我们实现最简单的贪婪搜索。


```python
def translate(encoder, decoder, input_seq, max_seq_len):
    """
    使用编码器-解码器模型进行序列翻译。

    Args:
    - encoder: 编码器模型对象。
    - decoder: 解码器模型对象。
    - input_seq (str): 要翻译的输入序列。
    - max_seq_len (int): 输出序列的最大长度限制。

    Returns:
    - output_tokens (list): 翻译后的输出序列的 token 列表。
    """
    
    # 分词输入序列并添加 EOS 和填充
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    
    # 将输入 tokens 转换为张量
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])  # batch=1
    
    # 初始化编码器状态
    enc_state = encoder.begin_state()
    
    # 编码器前向传播
    enc_output, enc_state = encoder(enc_input, enc_state)
    
    # 使用 BOS token 初始化解码器输入
    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    
    # 使用编码器状态初始化解码器状态
    dec_state = decoder.begin_state(enc_state)
    
    # 初始化列表以存储输出 tokens
    output_tokens = []
    
    # 解码最多 max_seq_len 个 token
    for _ in range(max_seq_len):
        # 解码器前向传播
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        
        # 预测具有最高概率的 token
        pred = dec_output.argmax(dim=1)
        
        # 将预测的 token 索引转换为 token 字符串
        pred_token = out_vocab.itos[int(pred.item())]
        
        # 检查是否为 EOS token
        if pred_token == EOS:
            break  # 如果预测到 EOS，则停止解码
        
        # 将预测的 token 添加到输出序列
        output_tokens.append(pred_token)
        
        # 更新解码器输入以准备下一个时间步
        dec_input = pred
    
    return output_tokens

```

简单测试一下模型。输入法语句子“ils regardent.”，翻译后的英语句子应该是“they are watching.”。


```python
input_seq = 'ils regardent .'
translate(encoder, decoder, input_seq, max_seq_len)
```




    ['they', 'are', 'watching', '.']



##  评价翻译结果

评价机器翻译结果通常使用BLEU（Bilingual Evaluation Understudy）[1]。对于模型预测序列中任意的子序列，BLEU考察这个子序列是否出现在标签序列中。

具体来说，设词数为$n$的子序列的精度为$p_n$。它是预测序列与标签序列匹配词数为$n$的子序列的数量与预测序列中词数为$n$的子序列的数量之比。举个例子，假设标签序列为$A$、$B$、$C$、$D$、$E$、$F$，预测序列为$A$、$B$、$B$、$C$、$D$，那么$p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0$。设$len_{\text{label}}$和$len_{\text{pred}}$分别为标签序列和预测序列的词数，那么，BLEU的定义为

$$ \exp\left(\min\left(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$

其中$k$是我们希望匹配的子序列的最大词数。可以看到当预测序列和标签序列完全一致时，BLEU为1。

因为匹配较长子序列比匹配较短子序列更难，BLEU对匹配较长子序列的精度赋予了更大权重。例如，当$p_n$固定在0.5时，随着$n$的增大，$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。另外，模型预测较短序列往往会得到较高$p_n$值。因此，上式中连乘项前面的系数是为了惩罚较短的输出而设的。举个例子，当$k=2$时，假设标签序列为$A$、$B$、$C$、$D$、$E$、$F$，而预测序列为$A$、$B$。虽然$p_1 = p_2 = 1$，但惩罚系数$\exp(1-6/2) \approx 0.14$，因此BLEU也接近0.14。

下面来实现BLEU的计算。



```python
import math
import collections

def bleu(pred_tokens, label_tokens, k):
    """
    计算 BLEU 分数以评估预测序列与参考序列之间的相似度。

    Args:
    - pred_tokens (list): 预测的 token 列表。
    - label_tokens (list): 参考的 token 列表。
    - k (int): 最大的 n-gram 计算阶数。

    Returns:
    - score (float): 计算得到的 BLEU 分数。
    """
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    
    # 长度惩罚项
    score = math.exp(min(0, 1 - len_label / len_pred))
    
    # 计算每个 n-gram 阶数从 1 到 k 的精度
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        
        # 统计参考序列中的 n-grams
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        
        # 统计预测序列中与参考序列匹配的 n-grams
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        
        # 计算当前 n-gram 阶数的精度
        precision = num_matches / (len_pred - n + 1)
        
        # 应用长度惩罚因子
        score *= math.pow(precision, math.pow(0.5, n))
    
    return score
```

接下来，定义一个辅助打印函数。


```python
def score(input_seq, label_seq, k):
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k),
                                      ' '.join(pred_tokens)))
```

预测正确则分数为1。


```python
score('ils regardent .', 'they are watching .', k=2)
```

    bleu 1.000, predict: they are watching .



```python
score('ils sont canadienne .', 'they are canadian .', k=2)
```

    bleu 0.658, predict: they are russian .


## 小结

* 可以将编码器—解码器和注意力机制应用于机器翻译中。
* BLEU可以用来评价翻译结果。


## 参考文献

[1] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.

[2] WMT. http://www.statmt.org/wmt14/translation-task.html

[3] Tatoeba Project. http://www.manythings.org/anki/

# Japanese-Chinese Machine Translation Model with Transformer & PyTorch

A tutorial using Jupyter Notebook, PyTorch, Torchtext, and SentencePiece

## Import required packages
Firstly, let’s make sure we have the below packages installed in our system, if you found that some packages are missing, make sure to install them.


```python
# 安装 torchtext 库
!pip install torchtext

# 导入 torchtext 库
import torchtext

# 打印 torchtext 版本
print(torchtext.__version__)

# 安装 torchtext 库
!pip install torchtext

# 安装 sentencepiece 库
!pip install sentencepiece

# 安装指定版本的 torchtext 库
!pip install torchtext==0.6.0

# 安装 pandas 库
!pip install pandas
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Requirement already satisfied: pandas in ./miniconda3/lib/python3.8/site-packages (2.0.3)
    Requirement already satisfied: numpy>=1.20.3 in ./miniconda3/lib/python3.8/site-packages (from pandas) (1.24.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in ./miniconda3/lib/python3.8/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in ./miniconda3/lib/python3.8/site-packages (from pandas) (2022.7.1)
    Requirement already satisfied: tzdata>=2022.1 in ./miniconda3/lib/python3.8/site-packages (from pandas) (2024.1)
    Requirement already satisfied: six>=1.5 in ./miniconda3/lib/python3.8/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m

实验环境

![image-20240623143608781](/Users/wangyang/Library/Application Support/typora-user-images/image-20240623143608781.png)

```python
import math
import torchtext
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import io
import time
import pandas as pd
import numpy as np
import pickle
import tqdm
import sentencepiece as spm
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.get_device_name(0)) ## 如果你有GPU，请在你自己的电脑上尝试运行这一套代码


```


```python
device
```


    device(type='cuda')



## Get the parallel dataset
In this tutorial, we will use the Japanese-English parallel dataset downloaded from JParaCrawl![http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl] which is described as the “largest publicly available English-Japanese parallel corpus created by NTT. It was created by largely crawling the web and automatically aligning parallel sentences.” You can also see the paper here.


```python
import pandas as pd

# 从指定路径读取 TSV 文件，并将其加载到 DataFrame 中
# 使用 \\t 作为分隔符，指定 engine 为 'python' 并且不使用任何列作为 header
df = pd.read_csv('./zh-ja.bicleaner05.txt', sep='\\t', engine='python', header=None)

# 提取 DataFrame 的第三列（索引为 2）并转换为列表，赋值给 trainen
trainen = df[2].values.tolist()  #[:10000]

# 提取 DataFrame 的第四列（索引为 3）并转换为列表，赋值给 trainja
trainja = df[3].values.tolist()  #[:10000]

# 移除 trainen 列表中的第 5972 个元素
# trainen.pop(5972)

# 移除 trainja 列表中的第 5972 个元素
# trainja.pop(5972)
```

After importing all the Japanese and their English counterparts, I deleted the last data in the dataset because it has a missing value. In total, the number of sentences in both trainen and trainja is 5,973,071, however, for learning purposes, it is often recommended to sample the data and make sure everything is working as intended, before using all the data at once, to save time.



Here is an example of sentence contained in the dataset.




```python
print(trainen[500])
print(trainja[500])
```

    Chinese HS Code Harmonized Code System < HS编码 2905 无环醇及其卤化、磺化、硝化或亚硝化衍生物 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...
    Japanese HS Code Harmonized Code System < HSコード 2905 非環式アルコール並びにそのハロゲン化誘導体、スルホン化誘導体、ニトロ化誘導体及びニトロソ化誘導体 HS Code List (Harmonized System Code) for US, UK, EU, China, India, France, Japan, Russia, Germany, Korea, Canada ...


We can also use different parallel datasets to follow along with this article, just make sure that we can process the data into the two lists of strings as shown above, containing the Japanese and English sentences.

## Prepare the tokenizers
Unlike English or other alphabetical languages, a Japanese sentence does not contain whitespaces to separate the words. We can use the tokenizers provided by JParaCrawl which was created using SentencePiece for both Japanese and English, you can visit the JParaCrawl website to download them, or click here.


```python
en_tokenizer = spm.SentencePieceProcessor(model_file='./spm.en.nopretok.model')
ja_tokenizer = spm.SentencePieceProcessor(model_file='./spm.ja.nopretok.model')
```

After the tokenizers are loaded, you can test them, for example, by executing the below code.




```python
en_tokenizer.encode("All residents aged 20 to 59 years who live in Japan must enroll in public pension system.")

```




    [227,
     2980,
     8863,
     373,
     8,
     9381,
     126,
     91,
     649,
     11,
     93,
     240,
     19228,
     11,
     419,
     14926,
     102,
     5]




```python


```


```python
ja_tokenizer.encode("年金 日本に住んでいる20歳~60歳の全ての人は、公的年金制度に加入しなければなりません。")
```




    [4,
     31,
     346,
     912,
     10050,
     222,
     1337,
     372,
     820,
     4559,
     858,
     750,
     3,
     13118,
     31,
     346,
     2000,
     10,
     8978,
     5461,
     5]



## Build the TorchText Vocab objects and convert the sentences into Torch tensors
Using the tokenizers and raw sentences, we then build the Vocab object imported from TorchText. This process can take a few seconds or minutes depending on the size of our dataset and computing power. Different tokenizer can also affect the time needed to build the vocab, I tried several other tokenizers for Japanese but SentencePiece seems to be working well and fast enough for me.


```python
from collections import Counter
from torchtext.vocab import Vocab

# 定义一个函数，用于构建词汇表
def build_vocab(sentences, tokenizer):
    # 创建一个计数器对象
    counter = Counter()
    # 遍历所有句子
    for sentence in sentences:
        # 使用分词器对句子进行编码并更新计数器
        counter.update(tokenizer.encode(sentence))
    # 创建词汇表对象，添加特殊标记
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# 使用给定的日语分词器构建日语词汇表
ja_vocab = build_vocab(trainja, ja_tokenizer)

# 使用给定的英语分词器构建英语词汇表
en_vocab = build_vocab(trainen, en_tokenizer)
```

After we have the vocabulary objects, we can then use the vocab and the tokenizer objects to build the tensors for our training data.




```python
import torch

# 定义一个函数，用于处理数据
def data_process(ja, en):
    data = []
    # 遍历日语和英语句子的元组
    for (raw_ja, raw_en) in zip(ja, en):
        # 使用日语词汇表将分词后的日语句子编码为张量
        ja_tensor_ = torch.tensor(
            [ja_vocab[token] for token in ja_tokenizer.encode(raw_ja.rstrip("\n"), out_type=str)],
            dtype=torch.long
        )
        # 使用英语词汇表将分词后的英语句子编码为张量
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer.encode(raw_en.rstrip("\n"), out_type=str)],
            dtype=torch.long
        )
        # 将编码后的张量对添加到数据列表中
        data.append((ja_tensor_, en_tensor_))
    # 返回处理后的数据
    return data

# 使用处理函数处理训练数据
train_data = data_process(trainja, trainen)

```

## Create the DataLoader object to be iterated during training
Here, I set the BATCH_SIZE to 16 to prevent “cuda out of memory”, but this depends on various things such as your machine memory capacity, size of data, etc., so feel free to change the batch size according to your needs (note: the tutorial from PyTorch sets the batch size as 128 using the Multi30k German-English dataset.)


```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# 定义批处理大小
BATCH_SIZE = 8

# 获取特殊标记的索引
PAD_IDX = ja_vocab['<pad>']
BOS_IDX = ja_vocab['<bos>']
EOS_IDX = ja_vocab['<eos>']

# 定义一个函数，用于生成批次数据
def generate_batch(data_batch):
    ja_batch, en_batch = [], []
    # 遍历数据批次中的每对句子
    for (ja_item, en_item) in data_batch:
        # 在日语句子的开头和结尾添加 BOS 和 EOS 标记，并将其加入批次
        ja_batch.append(torch.cat([torch.tensor([BOS_IDX]), ja_item, torch.tensor([EOS_IDX])], dim=0))
        # 在英语句子的开头和结尾添加 BOS 和 EOS 标记，并将其加入批次
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    # 对批次中的句子进行填充，使其具有相同长度
    ja_batch = pad_sequence(ja_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    # 返回填充后的批次数据
    return ja_batch, en_batch

# 创建数据加载器，用于生成训练数据的批次
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

```


## Sequence-to-sequence Transformer
The next couple of codes and text explanations (written in italic) are taken from the original PyTorch tutorial [https://pytorch.org/tutorials/beginner/translation_transformer.html]. I did not make any change except for the BATCH_SIZE and the word de_vocabwhich is changed to ja_vocab.

Transformer is a Seq2Seq model introduced in “Attention is all you need” paper for solving machine translation task. Transformer model consists of an encoder and decoder block each containing fixed number of layers.

Encoder processes the input sequence by propagating it, through a series of Multi-head Attention and Feed forward network layers. The output from the Encoder referred to as memory, is fed to the decoder along with target tensors. Encoder and decoder are trained in an end-to-end fashion using teacher forcing technique.


```python
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

# 定义一个类，用于序列到序列的 Transformer 模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()

        # 创建编码器层
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        # 创建 Transformer 编码器
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 创建解码器层
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        # 创建 Transformer 解码器
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 定义生成器线性层，将编码后的向量转换为目标词汇表的概率分布
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        # 定义源语言和目标语言的词嵌入层
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        
        # 定义位置编码层
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        # 对源语言句子进行词嵌入和位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # 对目标语言句子进行词嵌入和位置编码
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # 通过编码器对源语言句子进行编码
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        # 通过解码器对目标语言句子进行解码
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        # 生成最终的输出
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # 对源语言句子进行编码
        return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # 对目标语言句子进行解码
        return self.transformer_decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

```

Text tokens are represented by using token embeddings. Positional encoding is added to the token embedding to introduce a notion of word order.




```python
import math
import torch
from torch import nn, Tensor

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 计算位置编码的分母
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 生成位置序列
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 创建位置编码矩阵
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 将位置编码矩阵注册为缓冲区，防止其被优化器更新
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # 将位置编码添加到输入的 token embedding 上，并应用 Dropout
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])

# 定义 TokenEmbedding 类
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # 对输入 tokens 进行嵌入，并根据嵌入大小进行缩放
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

```

We create a subsequent word mask to stop a target word from attending to its subsequent words. We also create masks, for masking source and target padding tokens




```python
import torch

# 定义设备，通常是 'cuda' 或 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成一个上三角矩阵掩码，用于自回归解码器
def generate_square_subsequent_mask(sz):
    # 创建一个上三角矩阵，主对角线及其上方的元素为 1，其余为 0
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # 将 0 填充为 -inf，将 1 填充为 0.0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 创建源和目标的掩码，用于 Transformer 模型
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 生成目标序列的掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 源序列的掩码全为 0
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # 生成源和目标序列的填充掩码
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

```

Define model parameters and instantiate model. 这里我们服务器实在是计算能力有限，按照以下配置可以训练但是效果应该是不行的。如果想要看到训练的效果请使用你自己的带GPU的电脑运行这一套代码。

当你使用自己的GPU的时候，NUM_ENCODER_LAYERS 和 NUM_DECODER_LAYERS 设置为3或者更高，NHEAD设置8，EMB_SIZE设置为512。

设置超参数和模型参数，包括词汇表大小、嵌入维度、头数、前馈网络维度、批处理大小、编码器和解码器层数以及训练轮数。

初始化 `Seq2SeqTransformer` 模型，并使用 Xavier 均匀分布初始化模型参数。

将模型移动到指定设备（GPU 或 CPU）。

定义损失函数 `CrossEntropyLoss`，忽略填充标记的损失。

定义 Adam 优化器，并设置学习率和其他参数。

定义 `train_epoch` 函数，用于训练一个 epoch。它会对输入和目标进行掩码处理，计算损失并进行梯度更新。

定义 `evaluate` 函数，用于评估模型在验证集上的表现。它会计算验证集的平均损失。


```python
# 定义超参数和模型参数
SRC_VOCAB_SIZE = len(ja_vocab)
TGT_VOCAB_SIZE = len(en_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 16

# 初始化 Seq2Seq Transformer 模型
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

# 初始化模型参数
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# 将模型移动到指定设备（GPU 或 CPU）
transformer = transformer.to(device)

# 定义损失函数，忽略填充标记的损失
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 定义优化器
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

# 定义训练一个 epoch 的函数
def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)

# 定义评估函数
def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)

```

## Start training
Finally, after preparing the necessary classes and functions, we are ready to train our model. This goes without saying but the time needed to finish training could vary greatly depending on a lot of things such as computing power, parameters, and size of datasets.

When I trained the model using the complete list of sentences from JParaCrawl which has around 5.9 million sentences for each language, it took around 5 hours per epoch using a single NVIDIA GeForce RTX 3070 GPU.

Here is the code:


```python
for epoch in tqdm.tqdm(range(1, NUM_EPOCHS+1)):
  start_time = time.time()
  train_loss = train_epoch(transformer, train_iter, optimizer)
  end_time = time.time()
  print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s"))

```

      0%|          | 0/16 [00:00<?, ?it/s]/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched src_key_padding_mask and src_mask is deprecated. Use same type for both instead.
      warnings.warn(
    /root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
      warnings.warn(
      6%|▋         | 1/16 [06:26<1:36:32, 386.20s/it]
    
    Epoch: 1, Train loss: 5.088, Epoch time = 386.195s


     12%|█▎        | 2/16 [12:56<1:30:38, 388.47s/it]
    
    Epoch: 2, Train loss: 5.069, Epoch time = 390.053s


     19%|█▉        | 3/16 [19:26<1:24:19, 389.21s/it]
    
    Epoch: 3, Train loss: 5.068, Epoch time = 390.090s


     25%|██▌       | 4/16 [25:54<1:17:46, 388.89s/it]
    
    Epoch: 4, Train loss: 4.468, Epoch time = 388.397s


     31%|███▏      | 5/16 [32:21<1:11:08, 388.03s/it]
    
    Epoch: 5, Train loss: 4.268, Epoch time = 386.505s


     38%|███▊      | 6/16 [38:45<1:04:28, 386.82s/it]
    
    Epoch: 6, Train loss: 3.734, Epoch time = 384.463s


     44%|████▍     | 7/16 [45:13<58:03, 387.08s/it]  
    
    Epoch: 7, Train loss: 4.025, Epoch time = 387.606s


     50%|█████     | 8/16 [51:40<51:37, 387.17s/it]
    
    Epoch: 8, Train loss: 3.937, Epoch time = 387.375s


     56%|█████▋    | 9/16 [58:08<45:12, 387.51s/it]
    
    Epoch: 9, Train loss: 3.868, Epoch time = 388.239s


     62%|██████▎   | 10/16 [1:04:37<38:46, 387.78s/it]
    
    Epoch: 10, Train loss: 3.729, Epoch time = 388.387s


     69%|██████▉   | 11/16 [1:08:44<28:43, 344.61s/it]
    
    Epoch: 11, Train loss: 3.662, Epoch time = 246.728s


     75%|███████▌  | 12/16 [1:12:17<20:19, 304.76s/it]
    
    Epoch: 12, Train loss: 3.436, Epoch time = 213.606s


     81%|████████▏ | 13/16 [1:15:49<13:50, 276.73s/it]
    
    Epoch: 13, Train loss: 3.325, Epoch time = 212.219s


     88%|████████▊ | 14/16 [1:19:22<08:34, 257.50s/it]
    
    Epoch: 14, Train loss: 3.298, Epoch time = 213.063s


     94%|█████████▍| 15/16 [1:22:55<04:03, 243.91s/it]
    
    Epoch: 15, Train loss: 3.276, Epoch time = 212.429s


    100%|██████████| 16/16 [1:26:29<00:00, 324.33s/it]
    
    Epoch: 16, Train loss: 3.229, Epoch time = 213.948s


​    


## Try translating a Japanese sentence using the trained model
First, we create the functions to translate a new sentence, including steps such as to get the Japanese sentence, tokenize, convert to tensors, inference, and then decode the result back into a sentence, but this time in English.


```python
# 定义贪婪解码函数
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    # 编码源序列
    memory = model.encode(src, src_mask)
    # 初始化目标序列，起始符号
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        # 解码目标序列
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        # 通过生成器预测下一个单词的概率
        prob = model.generator(out[:, -1])
        # 选择概率最大的单词作为下一个单词
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        # 将下一个单词添加到目标序列
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# 定义翻译函数
def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    # 将源句子编码为 token 序列并添加起始和结束符号
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer.encode(src, out_type=str)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = torch.LongTensor(tokens).reshape(num_tokens, 1)
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    # 使用贪婪解码生成目标序列
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    # 将目标序列的 token 转换为单词，并移除起始和结束符号
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

```

Then, we can just call the translate function and pass the required parameters.




```python
translate(transformer, "HSコード 8515 はんだ付け用、ろう付け用又は溶接用の機器(電気式(電気加熱ガス式を含む。)", ja_vocab, en_vocab, ja_tokenizer)

```




    '_H S 用 于 _85 15_ 焊 、焊 接 设 备 、焊 接 电 气 式 （ 包 括 电 气 加 热 ) 。'




```python
trainen.pop(5)
```




    '美国 设施: 停车场, 24小时前台, 健身中心, 报纸, 露台, 禁烟客房, 干洗, 无障碍设施, 免费停车, 上网服务, 电梯, 快速办理入住/退房手续, 保险箱, 暖气, 传真/复印, 行李寄存, 无线网络, 免费无线网络连接, 酒店各处禁烟, 空调, 阳光露台, 自动售货机(饮品), 自动售货机(零食), 每日清洁服务, 内部停车场, 私人停车场, WiFi(覆盖酒店各处), 停车库, 无障碍停车场, 简短描述Gateway Hotel Santa Monica酒店距离海滩2英里(3.2公里),提供24小时健身房。每间客房均提供免费WiFi,客人可以使用酒店的免费地下停车场。'




```python
trainja.pop(5)
```




    'アメリカ合衆国 施設・設備: 駐車場, 24時間対応フロント, フィットネスセンター, 新聞, テラス, 禁煙ルーム, ドライクリーニング, バリアフリー, 無料駐車場, インターネット, エレベーター, エクスプレス・チェックイン / チェックアウト, セーフティボックス, 暖房, FAX / コピー, 荷物預かり, Wi-Fi, 無料Wi-Fi, 全館禁煙, エアコン, サンテラス, 自販機(ドリンク類), 自販機(スナック類), 客室清掃サービス(毎日), 敷地内駐車場, 専用駐車場, Wi-Fi(館内全域), 立体駐車場, 障害者用駐車場, 短い説明Gateway Hotel Santa Monicaはビーチから3.2kmの場所に位置し、24時間利用可能なジム、無料Wi-Fi付きのお部屋、無料の地下駐車場を提供しています。'




```python

```

## Save the Vocab objects and trained model
Finally, after the training has finished, we will save the Vocab objects (en_vocab and ja_vocab) first, using Pickle.


```python
import pickle
# open a file, where you want to store the data
file = open('en_vocab.pkl', 'wb')
# dump information to that file
pickle.dump(en_vocab, file)
file.close()
file = open('ja_vocab.pkl', 'wb')
pickle.dump(ja_vocab, file)
file.close()
```

Lastly, we can also save the model for later use using PyTorch save and load functions. Generally, there are two ways to save the model depending what we want to use them for later. The first one is for inference only, we can load the model later and use it to translate from Japanese to English.




```python
# save model for inference
torch.save(transformer.state_dict(), 'inference_model')
```

The second one is for inference too, but also for when we want to load the model later, and want to resume the training.




```python
# save model + checkpoint to resume training later
torch.save({
  'epoch': NUM_EPOCHS,
  'model_state_dict': transformer.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'loss': train_loss,
  }, 'model_checkpoint.tar')
```

# Conclusion
That’s it!

