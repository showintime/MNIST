#!/usr/bin/env python
# coding: utf-8



#Import chunk


import numpy as np

import math

import struct


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
    




def show_train(index):
    print('label:'+str(train_label[index]))
    a_image=train_image[index].reshape(-1,28)
    plt.imshow(a_image,cmap='gray')
    
    
def show_validate(index):
    print('label:'+str(validate_label[index]))
    a_image=validate_image[index].reshape(-1,28)
    plt.imshow(a_image,cmap='gray')
    
    
def show_test(index):
    print('label:'+str(test_label[index]))
    a_image=test_image[index].reshape(-1,28)
    plt.imshow(a_image,cmap='gray')



label_keys=np.identity(dimension[-1])



def loss(image,label,parameters):
    y_predict=predict(image,parameters)
    y_label=label_keys[label]
    difference=y_predict-y_label
    return np.dot(difference,difference)





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
    
  
    
    
    zero_grad=init_zero_grad()
    
    
    #calculate grad b0    
    for i in range(784):
        #对于b0的每一个分量b0[i]
        ctem=0
        for j in range(10):
            tem=2*(l3[j]-label_keys[label][j])
            tem2=0
            for k in range(10):
                tem1=parameters[1]['w'][i][k]*(1-(np.tanh(h[i]))**2)
                if k==j:
                    tem1*=(l3[k]-l3[k]**2)
                else:
                    tem1*=(-l3[k]*l3[j])
                    
                tem2+=tem1
            
            tem*=tem2
            
            ctem+=tem
            
        zero_grad[0]['b'][i]+=ctem
    
    #calculate grad w1
    for i in range(784):
        for j in range(10):
            #对于w1的每一个分量w1[i][j]
            ctem=0
            for k in range(10):
                tem=2*(l3[k]-label_keys[label][k])*h[i]
                if k==j:
                    tem*=(l3[k]-l3[k]**2)
                else:
                    tem*=(-l3[k]*l3[j])
                ctem+=tem 
            zero_grad[1]['w'][i][j]+=ctem 
        
    #calculate grad b1
    for i in range(10):     
        #对于b1的每一个分量b1[i]
        ctem=0
        for k in range(10):
            
            tem=2*(l3[k]-label_keys[label][k])
            if i==k:
                tem*=(l3[k]-l3[k]**2)
            else:
                tem*=(-l3[i]*l3[k])
            ctem+=tem
            
        zero_grad[1]['b'][i]+=ctem
            
    #print(zero_grad)
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





parameters=init_wb()





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







step=0.0001

def validate_grad(image,label):
    
    l0=image+parameters[0]['b']
    
    h=activation[0](l0)
    
    l2=np.dot(h,parameters[1]['w'])+parameters[1]['b']
    
    l3=activation[1](l2)
    
    exp_l2=np.exp(l2)
    
    
    #print('h:')
    #print(h)
    
    print('l2:')
    print(l2)
    print('l3:')
    print(l3)
    print('l3_sum:'+str(l3.sum()))
    print('label:')
    print(label_keys[label])
    print('exp_l2:')
    print(exp_l2)
    print('exp_l2_sum:')
    print(exp_l2.sum())
    
    zero_grad=init_zero_grad()
    
    for ii in range(len(zero_grad)):
        for jj in zero_grad[ii].keys():
            zero_grad[ii][jj]=zero_grad[ii][jj]+parameters[ii][jj]

            
    loss0=loss(image,label,parameters)
    #验证w1参数
    for i in range(784):
        
        for j in range(10):  
            
            zero_grad=init_zero_grad()
            for ii in range(len(zero_grad)):
                for jj in zero_grad[ii].keys():
                    zero_grad[ii][jj]=zero_grad[ii][jj]+parameters[ii][jj]

            
            zero_grad[1]['w'][i][j]=zero_grad[1]['w'][i][j]+step
            loss1=loss(image,label,zero_grad)
            
            #print('loss0='+str(loss0))
            #print('loss1='+str(loss1))
            
            calc_grid=0
            
            tem=0
            
            for k in range(10):
                
                if j==k:
                    tem=(softmax(l2)[j]-(softmax(l2)[j])**2)
                else:
                    tem=(-softmax(l2)[j]*softmax(l2)[k])
                
                #print('Rem1:tem='+str(tem))
                tem=tem*2*(l3[k]-label_keys[label][k])*h[i]
                #print('Rem2:tem='+str(tem))
                calc_grid=calc_grid+tem
            print('w1'+str([i])+str([j]),end='')
            print('计算梯度为：'+str(calc_grid),end='')
            print('差商梯度为：'+str((loss1-loss0)/step))
    






def test_grad_correct():
    step=0.0001

    test_input=np.random.rand(4)
    print(test_input)
    v0=softmax(test_input)
    print(v0)

    test_grad=[]
    test_input_exp=np.exp(test_input-test_input.max())
    test_input_exp_sum=test_input_exp.sum()
    
    
    
    for i in range(4):
        if i==1:
            test_grad.append(test_input_exp[i]/test_input_exp_sum-(test_input_exp[i]/test_input_exp_sum)**2)
        else:
            test_grad.append(-test_input_exp[i]*test_input_exp[1]/(test_input_exp_sum**2))
    


    test_input[1]=test_input[1]+step
    v1=softmax(test_input)
    print(v1)
    print('差商：'+str((v1-v0)/step))
    print('梯度：'+str(test_grad))









def write_parameters():

    f=open('111.txt','w')
    
    
    
    
    f.write('第1层：\n')
    f.write('b:\n')
    for i in parameters[0]['b']:
        f.write(str(i)+'\t')
    f.write('\n第2层：\n')
    f.write('b:\n')
    for i in parameters[1]['b']:
        f.write(str(i)+'\t')
    f.write('\nw:\n')
    for i in parameters[1]['w']:
        for j in i:
            f.write(str(j)+'\t')
        f.write('\n')
    f.write('\n')
    f.close()




def rrrr(index):
    print('预测序号：{}'.format(index))
    print('我的预测：{}'.format(predict(train_image[index],parameters).argmax()))
    show_train(index)








def train_verify(parameters):
    error_num=0
    for i in range(train_num):
        
        if predict(train_image[i],parameters).argmax()!=train_label[i]:
        
            error_num+=1
            
    rate=1-error_num/train_num
    print('准确率：'+str(rate*100)+'%')

    



time0=time.time()
for i in range(100):
    calculate_a_grad(train_image[0],train_label[0])
time1=time.time()
print((time1-time0)/100)
