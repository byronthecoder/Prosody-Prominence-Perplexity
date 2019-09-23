import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from collections import Counter
import csv
from matplotlib import pyplot as plt
import time

torch.random.manual_seed(0)
np.random.seed(0)

######################################################################
# data processing
with open('/Users/the-imitation-gamer/Documents/SLP/Msc_Dissertation/Prosody-and-Perplexity/wimp_3.pkl', 'rb') as f:
    data = pickle.load(f)
# with open('./words.pkl', 'rb') as f:
#     words = np.array(pickle.load(f))

train_data = data[: 22250]
test_data = data[22250:]
# unique_word = set(words[0])

# for i in range(len(words)-1):
#     if words[i+1] in unique_word:
#         test_data.append(data[i])
#     else:
#         train_data.append(data[i])
#         unique_word.add(words[i+1])
#

# BIGRAM_NUM = 23
LABEL_NUM = 3
FEATURE_TYPE = 11


######################################################################
# neural network model define
class SimpleModel(nn.Module):
    def __init__(self, hid1_dim=32):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(FEATURE_TYPE, hid1_dim)
        self.linear2 = nn.Linear(hid1_dim, LABEL_NUM)

    def forward(self, x):
        # x is vector which concat the representation of a word
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        log_probs = F.log_softmax(x, dim=1)

        return log_probs


######################################################################
# model optimization setting
learning_rate = 0.05  # 0.05, 0.01
max_epoch_num = 100

loss_function = nn.NLLLoss()
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


######################################################################
# train and test modules
def train():
    model.train()

    loss = torch.Tensor([0])
    correct_pred = 0
    model.zero_grad()

    np.random.shuffle(train_data)
    for vector, label in train_data:
        log_probs = model(torch.Tensor([vector.tolist()]))
        # loss += loss_function(log_probs, torch.autograd.Variable(torch.LongTensor([int(label-1)]))) # word index
        loss += loss_function(log_probs, torch.autograd.Variable(torch.LongTensor([int(label)])))
        pred = torch.argmax(log_probs)
        # if word[pred] == words[label]:
        if pred == label:
            correct_pred += 1
            # return correct_pred, loss

    # print('train: ', correct_pred/len(train_data))
    # print('train loss: ', loss)
    loss.backward()
    optimizer.step()
    accuracy = correct_pred/len(train_data)
    return accuracy, loss


def test():
    model.eval()

    correct_pred = 0
    model.zero_grad()

    for vector, label in test_data:
        log_probs = model(torch.Tensor([vector.tolist()]))
        pred = torch.argmax(log_probs)
        # if words[pred] == words[label]:
        if pred == label:
            correct_pred += 1

    accuracy = correct_pred / len(test_data)
    return accuracy

    # print('test', correct_pred/len(test_data))


######################################################################
# train and test modules
print('training begins...')
eps = []
accuracy_train_l = []
accuracy_test_l = []
loss_l = []

t0 = time.time()
for ep in range(max_epoch_num):
    accuracy_train, loss = train()
    accuracy_test = test()

    if ep % 10 == 0:
        t1 = time.time()
        print('\nepoch: ', ep)
        print('train accuracy: ', accuracy_train)
        print('train loss: ', loss)
        print('test', accuracy_test)
        print('time cost: ', t1-t0)
        t0 = t1

        eps.append(ep)
        accuracy_train_l.append(accuracy_train)
        accuracy_test_l.append(accuracy_test)
        loss_l.append(loss)

# plot
plt.figure()
plt.plot(eps, accuracy_train_l)
plt.plot(eps, accuracy_test_l)
# titile
plt.title("3 label epoch num" + str(max_epoch_num))
# x/y axis label
plt.xlabel('epochs')
plt.ylabel('accuracy')
# legend
plt.legend(['train', 'test'])
# show plot
plt.show()

plt.figure()
plt.plot(eps, loss_l)
plt.title("learning rate: " + str(learning_rate))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()







