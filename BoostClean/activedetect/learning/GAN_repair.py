'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import numpy as np
from tqdm import tqdm

# import pytorch
import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F # 各中层函数的实现，与层类型对应
import torch.optim as optim	# 实现各种优化算法的包
from torch.autograd import Variable

from activedetect.learning.utils import normalization, renormalization, rounding
from activedetect.learning.utils import xavier_init
from activedetect.learning.utils import binary_sampler, uniform_sampler, sample_batch_index
from activedetect.learning.networks_gan_repair import DiscriminatorNet, GeneratorNet, ClassifierNet, Discriminator_loss, Generator_loss, Classifier_loss

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def gain (data_x, gain_parameters, labels, zero_index):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = np.ones_like(data_x, dtype=float)
  for i,j in zero_index:
    data_m[i][j] = 0

  label_encoder = LabelEncoder()
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = label_encoder.fit_transform(labels)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  labels = onehot_encoder.fit_transform(integer_encoded)
  # labels = np.array(labels)
  labels = labels.astype(np.float)

  # data_m_0 = 1-np.isnan(data_m_1)
  # data_m = np.concatenate((data_m_1, data_m_0), axis=1)
  data_x = np.array(data_x, dtype=float)

  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']

  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)

  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  # norm_data_x = np.nan_to_num(norm_data, 0)
  norm_data_x = norm_data

  label = []

  GNet = GeneratorNet(dim*2, dim, dim)
  DNet = DiscriminatorNet(dim*2, dim, dim)
  CNet = ClassifierNet(dim, dim, 2)

  G_optimizer = torch.optim.Adam(GNet.parameters(), lr=0.001)
  D_optimizer = torch.optim.Adam(DNet.parameters(), lr=0.001)
  C_optimizer = torch.optim.Adam(CNet.parameters(), lr=0.001)
  
  G_Loss = Generator_loss()  # this is for regression mean squared loss
  D_Loss = Discriminator_loss()
  C_Loss = torch.nn.CrossEntropyLoss()

  #初始化
  # D_prediction = torch.rand_like(data_m)
  # D_prediction = torch.round(D_input)

  for it in tqdm(range(iterations)):
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    labels_mb = labels[batch_idx]
    M_mb = data_m[batch_idx, :]
    
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    X_mb = torch.from_numpy(X_mb).float()
    H_mb = torch.from_numpy(H_mb).float()
    Z_mb = torch.from_numpy(Z_mb).float()
    M_mb = torch.from_numpy(M_mb).float()
    labels_mb = torch.from_numpy(labels_mb).float()
    # labels_mb = torch.reshape(labels_mb, (-1,1))



    #Generator 网络训练
    #传播网络
    D_prediction_temp = DNet(torch.cat((X_mb, H_mb), 1)).detach()
    D_prediction_temp = torch.round(D_prediction_temp)

    G_input = torch.cat((X_mb, D_prediction_temp), 1)
    G_prediction = GNet(G_input)

    repair_temp = X_mb * D_prediction_temp + G_prediction * (1-D_prediction_temp)
    Classify_res_temp = CNet(repair_temp)

    C_optimizer.zero_grad()
    G_optimizer.zero_grad()

    loss_C = C_Loss(Classify_res_temp, torch.max(labels_mb, 1)[1])
    loss_C.backward()

    #更新梯度
    C_optimizer.step()
    G_optimizer.step()


    #Discriminator网络训练
    #传播三个网络
    repair_temp = repair_temp.detach()

    D_input = torch.cat((repair_temp,  H_mb), 1)
    D_prediction = DNet(D_input)

    D_optimizer.zero_grad()

    loss_D = D_Loss(D_prediction, M_mb)
    loss_D.backward(retain_graph=True)

    D_optimizer.step()


  ## Return imputed data      
  # Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  # X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

  X_mb = torch.from_numpy(X_mb).float()
  # H_mb = torch.from_numpy(H_mb).float()
  # Z_mb = torch.from_numpy(Z_mb).float()
  M_mb = torch.from_numpy(M_mb).float()

  repair_data = GNet(torch.cat((X_mb, M_mb), 1))
  repair_data = repair_data.detach().numpy()
  repair_data = data_m * norm_data_x + (1-data_m) * repair_data
  
  # Renormalization
  repair_data = renormalization(repair_data, norm_parameters)  
  
  # Rounding
  repair_data = rounding(repair_data, data_x)  
          
  return repair_data