
import numpy as np

import math

import struct

import random

import time


import matplotlib.pyplot as plt




def tanh(x):
    return np.tanh(x)

def softmax(x):
    #这里减去x.max()为了防止指数爆炸，并不会影响结果
    exp=np.exp(x-x.max())
    return exp/exp.sum()





#Initial chunk

#两层的维度
dimension=[784,10]
#两层的激活函数
activation=[tanh,softmax]

distribution=[
    {'b':[0,0]},
    {'b':[0,0],'w':[-math.sqrt(6/(dimension[0]+dimension[1])),math.sqrt(6/(dimension[0]+dimension[1]))]}
]




#Parameter's initial


def init_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimension[layer])*(dist[1]-dist[0])+dist[0]


def init_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimension[layer-1],dimension[layer])*(dist[1]-dist[0])+dist[0]

def init_wb():
    parameter=[]
    for i in range(len(distribution)):
        layer_parameter={}
        for j in distribution[i].keys():
            if j=='w':
                layer_parameter['w']=init_w(i)
            elif j=='b':
                layer_parameter['b']=init_b(i)
        parameter.append(layer_parameter)
    return parameter


parameters=init_wb()



def predict(image,parameters):
    hidden=activation[0](image+parameters[0]['b'])
    output=activation[1](np.dot(hidden,parameters[1]['w'])+parameters[1]['b'])
    return output




data_path='./datas/'

train_image_path=data_path+'train-images.idx3-ubyte'
train_label_path=data_path+'train-labels.idx1-ubyte'

test_image_path=data_path+'t10k-images.idx3-ubyte'
test_label_path=data_path+'t10k-labels.idx1-ubyte'












train_num=50000
validate_num=10000
test_num=10000

with open(train_image_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    tem_image=np.fromfile(f,dtype=np.uint8).reshape(-1,784)
    train_image=tem_image[:train_num]
    validate_image=tem_image[train_num:]
    
with open(test_image_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    test_image=np.fromfile(f,dtype=np.uint8).reshape(-1,784)
    

with open(train_label_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    tem_label=np.fromfile(f,dtype=np.uint8)
    train_label=tem_label[:train_num]
    validate_label=tem_label[train_num:]

with open(test_label_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    test_label=np.fromfile(f,dtype=np.uint8)    
    






label_keys=np.identity(dimension[-1])







def validate_verify(parameters):
    error_num=0
    for i in range(validate_num):
        
        if predict(validate_image[i],parameters).argmax()!=validate_label[i]:
        
            error_num+=1
            
    rate=1-error_num/validate_num
    print('准确率：'+str(rate*100)+'%')
    



def init_zero_grad():
    parameter=[]
    for layer in range(len(distribution)):
        layer_parameter={}
        for j in distribution[layer].keys():
            if j=='w':
                layer_parameter['w']=np.zeros((dimension[layer-1],dimension[layer]))
            elif j=='b':
                layer_parameter['b']=np.zeros(dimension[layer])
        parameter.append(layer_parameter)
    return parameter





def calculate_a_grad(image,label):
    
    l0=image+parameters[0]['b']
    
    h=activation[0](l0)
    
    l2=np.dot(h,parameters[1]['w'])+parameters[1]['b']
    
    l3=activation[1](l2)
    
  
    exp_l2=np.exp(l2-l2.max())
    
    zero_grad=init_zero_grad()
    
    
    #calculate grad b0    
    for i in range(784):
        #对于b0的每一个分量b0[i]
        for j in range(10):
            tem=2*(l3[j]-label_keys[label][j])
            tem2=0
            for k in range(10):
                tem1=parameters[1]['w'][i][k]*(1-(np.tanh(h[i]))**2)
                if k==j:
                    tem1=tem1*(l3[k]-l3[k]**2)
                else:
                    tem1=tem1*(-l3[k]*l3[j])
                    
                tem2+=tem1
            
            tem*=tem2
            
            zero_grad[0]['b'][i]+=tem
    
    #calculate grad w1
    for i in range(784):
        for j in range(10):
            #对于w1的每一个分量w1[i][j]
            for k in range(10):
                tem=2*(l3[k]-label_keys[label][k])*h[i]
                if k==j:
                    tem*=(l3[k]-l3[k]**2)
                else:
                    tem*=(-l3[k]*l3[j])
                    
                zero_grad[1]['w'][i][j]+=tem 
        
    #calculate grad b1
    for i in range(10):     
        #对于b1的每一个分量b1[i]
        for k in range(10):
            
            tem=2*(l3[k]-label_keys[label][k])
            if i==k:
                tem*=(l3[k]-l3[k]**2)
            else:
                tem*=(-l3[i]*l3[k])
                  
            zero_grad[1]['b'][i]+=tem
            
    
    return zero_grad
            
            




banch_num=500
def calculate_banch_grad(banch_time):
    
    banch_grad=init_zero_grad()
    for i in range(banch_num*banch_time,banch_num*(banch_time+1)):
        
        #print('训练'+str(i+1)+'/'+str(banch_num)+'个参数')
        tem_grad=calculate_a_grad(train_image[i],train_label[i])
        for j in range(len(banch_grad)):
            for k in banch_grad[j].keys():
                banch_grad[j][k]+=tem_grad[j][k]
    for j in range(len(banch_grad)):
        for k in banch_grad[j].keys():
            banch_grad[j][k]/=banch_num
    
    
    return banch_grad           






def train_banch(banch_grad,learn_rate):
    print('正在修正参数')
    for j in range(len(banch_grad)):
        for k in banch_grad[j].keys():
            parameters[j][k]-=learn_rate*banch_grad[j][k]
            
            






learn_rate=1
def train():
    
    for i in range(89,100):
        time0=time.time()
        print('#===============开始训练第'+str(i+1)+'/'+str(train_num//banch_num)+'组==============#')
        banch_grad=calculate_banch_grad(i)
        train_banch(banch_grad,learn_rate)
        time1=time.time()
        print('当前组用时：{}s'.format(time1-time0))
        validate_verify(parameters)
    print('Train Over!')
    
    



def trains():
	tem_time0=time.time()
	for i in range(1):
		train()
	tem_time1=time.time()
	print('总用时：{}'.format(tem_time1-tem_time0))








