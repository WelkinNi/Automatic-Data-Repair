# import pytorch
import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F # 各中层函数的实现，与层类型对应
import torch.optim as optim	# 实现各种优化算法的包
from torch.autograd import Variable

class GeneratorNet(torch.nn.Module): 
  def __init__(self, n_feature, n_hidden, n_output): 
    super(GeneratorNet, self).__init__() 
    self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
    self.hidden2 = torch.nn.Linear(n_hidden, n_hidden) 
    self.predict = torch.nn.Linear(n_hidden, n_output)
  
  def forward(self, input): 
    # print(input.shape)

    out = self.hidden1(input)
    out = F.relu(out)
    out = self.hidden2(out)
    out = F.relu(out)
    out = self.predict(out)
    out = F.sigmoid(out)
    return out

class DiscriminatorNet(torch.nn.Module): 
  def __init__(self, n_feature, n_hidden, n_output): 
    super(DiscriminatorNet, self).__init__() 
    self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
    self.hidden2 = torch.nn.Linear(n_hidden, n_hidden) 
    self.predict = torch.nn.Linear(n_hidden, n_output)
  
  def forward(self, input): 
    # print(input.shape)
    # print(self.hidden1)
    out = self.hidden1(input)
    out = F.relu(out)
    out = self.hidden2(out)
    out = F.relu(out)
    out = self.predict(out)
    out = F.sigmoid(out)
    return out

# class ClassifierNet(torch.nn.Module): 
#   def __init__(self, n_feature, n_hidden, n_output): 
#     super(ClassifierNet, self).__init__() 
#     self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
#     self.hidden2 = torch.nn.Linear(n_hidden, n_hidden) 
#     self.predict = torch.nn.Linear(n_hidden, n_output)
  
#   def forward(self, input): 
#     out = self.hidden1(input)
#     out = torch.sigmoid(out)
#     out = self.hidden2(out)
#     out = torch.sigmoid(out)
#     out = self.predict(out)
#     out = torch.sigmoid(out)
#     return out

class ClassifierNet(torch.nn.Module): 
    def __init__(self,n_input,n_hidden,n_output):
      super(ClassifierNet,self).__init__()
      self.hidden1 = torch.nn.Linear(n_input,n_hidden)
      # self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
      self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        # out = self.hidden2(out)
        # out = F.sigmoid(out)
        out = self.predict(out)
        # out = F.softmax(out)
        return out

class Discriminator_loss(nn.Module):
  def __init__(self):
    super().__init__()
        
  def forward(self, predict, real_M):
    loss = -torch.mean(real_M * torch.log(predict + 1e-8) + (1-real_M) * torch.log(1. - predict + 1e-8)) 
    return loss

class Generator_loss(nn.Module):
    def __init__(self):
      super().__init__()
        
    def forward(self, X, D_prob, G_sample, M, alpha):
      G_loss_temp = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
      MSE_loss = torch.mean((M * X - M * G_sample)**2) / torch.mean(M)
      return G_loss_temp + alpha * MSE_loss 


class Classifier_loss(nn.Module):
  def __init__(self):
    super().__init__()
        
  def forward(self, predict, real_M):
    loss = -torch.mean(real_M * torch.log(predict + 1e-8) + (1-real_M) * torch.log(1. - predict + 1e-8)) 
    return loss
