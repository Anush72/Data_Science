from DeepLeaningFramework import *
import sys,random,math
from collections import Counter
import numpy as np

# Opening the file
with open('qa1_single-supporting-fact_train.txt','r') as f:
    raw = f.readlines()

# Creating the encoding using tokenizations
tokens = [line.lower().replace('\n','').replace('\t','').split(" ")[1:]for line in raw[0:1000] ]
new_tokens = [(['-']*(6-len(line))+line) for line in tokens]
tokens = new_tokens
vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i

def word2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx
indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)
data = np.array(indices)

embed = Embedding(vocab_size=len(vocab),dim=16)
model = RNNCell(n_inputs=16,n_hidden=16,n_output=len(vocab))
criterion = CrossEntropyLoss()
params = model.get_parameters() + embed.get_parameters()
optim = SGD(parameters=params,alpha=0.005)
for iter in range(1000):
    batch_size = 100
    total_loss = 0
    hidden = model.init_hidden(batch_size=batch_size)
    for t in range(5):
        input = Tensor(data[0:batch_size,t],autograd=True)
        rnn_input = embed.forward(input=input)
        output,hidden = model.forward(input=rnn_input,hidden=hidden)
    target = Tensor(data[0:batch_size,t+1],autograd=True)
    loss = criterion.forward(output,target)
    loss.backward()
    optim.step()
    total_loss += loss.data
    if (iter%200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print_loss = total_loss  / (len(data)/batch_size)
        print(f'Loss: {print_loss}, Correct: {p_correct}')

batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = Tensor(data[0:batch_size,t], autograd=True)
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size,t+1], autograd=True)
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context:",ctx)
print("Pred:", vocab[output.data.argmax()])