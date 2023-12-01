import torch
from activedetect.learning.networks_gan_repair import DiscriminatorNet, GeneratorNet, ClassifierNet, Discriminator_loss, Generator_loss, Classifier_loss
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
# print(x0)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)


x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1)).type(torch.LongTensor)
# print(y)
x,y = Variable(x),Variable(y)

net = Net(2,20,2)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net(x)
    # print(prediction)
    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

prediction = torch.max(F.softmax(out),1)[1]
pred_y = prediction.data.numpy().squeeze()
target_y = y.data.numpy()
print (classification_report(target_y, pred_y))
