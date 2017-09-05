'''
Description: SMO
@author: huilin
@version:1

'''

import sys
import os
import numpy as np
import random

K=[]
xPro=[]
X=[]
Y=[]
test=0
train_num=1000
gamma=1.0/(2*5**2)
alpha = np.zeros((1,train_num))[0]
error_cache=np.zeros((1,train_num))[0]
b=0.0
C=1.0
tolerance=0.001
eps=1.0e-12

def get_xy(filename,data_num,offset):
    global X,Y
    matrix=np.loadtxt(filename,delimiter=',')
    '''
    f = open(filename,'r')
    lines = f.readlines()#get a list, each of its component is a line
    for i in range(offset,offset+data_num):
        matrix.append(lines[i].split(','))#add element
    matrix=np.array(matrix)
    '''
    matrix=np.hsplit(matrix,np.array([4]))
    X=matrix[0]
    Y=matrix[1]

def eta_f(i,j):
    return 2*K[i][j]-K[i][i]-K[j][j]

def obj(j):
    fx=0.0
    for i in range(0,train_num):
        fx +=alpha[i]*Y[i]*K[i][j]
    return fx-b

def xProduct():
    global xPro, K
    xPro=np.zeros((train_num,train_num))
    K=np.zeros((train_num,train_num))
    for i in range(0,train_num):
        for j in range(0,train_num):
            xPro[i][j]=np.dot(X[i],X[j])
            K[i][j]=np.exp(-gamma*xPro[i][j])


def error_Cache(i,j,alpha1new,alpha2newclipped,delta_b):
    t1=Y[i]*(alpha1new-alpha[i])
    t2=Y[j]*(alpha2newclipped-alpha[j])
    for k in range(0,train_num):
        if(alpha[k]>0 and alpha[k]<C):
            K_ik=K[i][k]
            K_jk=K[j][k]
            error_cache[k]+=t1*K_ik+t2*K_jk-delta_b
    error_cache[i]=error_cache[j]=0


def Ei(fx,y):
    return fx-y

def L_f(alpha1,alpha2,y1,y2):
    if(y1!=y2):
        if (0 >= alpha2-alpha1): return 0
        else:  return alpha2-alpha1
    elif (0>=alpha2+alpha1-C): return 0
    else: return alpha2+alpha1-C

def H_f(alpha1,alpha2,y1,y2):
    if(y1!=y2):
        if (C>=C+alpha2-alpha1): return C+alpha2-alpha1
        else: return C
    elif (C>=alpha2+alpha1): return alpha2+alpha1
    else: return C

def renew_alpha_b(i,j,Ei,Ej):
    global alpha,b
    alpha1=alpha[i]
    alpha2=alpha[j]
    L = L_f(alpha1,alpha2,Y[i],Y[j])
    H = H_f(alpha1,alpha2,Y[i],Y[j])
    eta = eta_f(i,j)
    if(L==H): return 0
    alpha2positive=alpha2-Y[i]*(Ei-Ej)/eta
    s=Y[i]*Y[j]
    if eta>0.001:
        alpha2new=alpha2positive
        if(alpha2new>=H): alpha2newclipped=H
        elif(alpha2new<=L): alpha2newclipped=L
        else: alpha2newclipped=alpha2new
    else:
        func1=Y[i]*(Ei+b)-alpha1*K[i][i]-s*alpha2*K[i][j]
        func2=Y[j]*(Ej+ b) - s*alpha1*K[i][j] - alpha2*K[j][j]
        L1=alpha1+s*(alpha2-L)
        H1=alpha1+s*(alpha2-H)
        Lobj = L1*func1+L*func2+1/2*L1*L1*K[i][i]+1/2*L1*L1*K[j][j]+s*L*L1*K[i][j]
        Hobj = H1*func1+H*func2+1/2*H1*H1*K[i][i]+1/2*H1*H1*K[j][j]+s*H*L1*K[i][j]
        if (Lobj>Hobj+eps): alpha2newclipped=L
        elif (Lobj<Hobj-eps): alpha2newclipped=H
        else: alpha2newclipped = alpha2
    if(np.fabs(alpha2-alpha2newclipped)<eps): return 0
    alpha1new=alpha1+s*(alpha2-alpha2newclipped)

    if (alpha1new>0 and alpha1new<C):
        bnew = Ei+Y[i]*(alpha1new-alpha1)*K[i][i]+Y[j]*(alpha2newclipped-alpha2)*K[i][j]+b
    else:
        if((alpha2newclipped>0 and alpha2newclipped<C)):
            bnew=Ej+Y[i]*(alpha1new-alpha1)*K[i][j]+Y[j]*(alpha2newclipped-alpha2)*K[j][j]+b
        else:
            b1=Ei+Y[i]*(alpha1new-alpha1)*K[i][i]+Y[j]*(alpha2newclipped-alpha2)*K[i][j]+b
            b2=Ej+Y[i]*(alpha1new-alpha1)*K[i][j]+Y[j]*(alpha2newclipped-alpha2)*K[j][j]+b
            bnew=(b1+b2)/2
    delta_b=bnew-b;
    alpha[i]=alpha1new
    alpha[j]=alpha2newclipped
    b=bnew
    error_Cache(i,j,alpha1new,alpha2newclipped,delta_b)
    return 1

#takeStep : optimize alpha1,alpha2
def takeStep(i,j):
    if(i==j): return 0
    Ei=Ej=0.0
    alpha1=alpha[i]
    alpha2=alpha[j]
    yi=Y[i]
    yj=Y[j]
    if(alpha1>0 and alpha1<C): Ei=error_cache[i]
    else: Ei=obj(i)-yi
    if(alpha2>0 and alpha2<C): Ej=error_cache[j]
    else: Ej=obj(j)-yj
    #renew alpha, b
    if renew_alpha_b(i,j,Ei,Ej): return 1
    else: return 0

#return the first multiplier alpha1
def examineExample(i):
    global test
    if(alpha[i]>0 and alpha[i]<C): Ei = error_cache[i]
    else: Ei=obj(i)-Y[i]
    r1=Y[i]*Ei
    if((r1>tolerance and alpha[i]>0) or (r1<-tolerance  and  alpha[i]<C)):
        #Use 3 methods to choose the second multiplier
        if(examFirstChoice(i,Ei)): return 1
        if(examNonBound(i)): return 1
        if(examBound(i)): return 1
    #test+=1
    #print "testtimes:", test
    return 0

#1.find maximumfabs(E1-E2) from non_bound
def examFirstChoice(i,Ei):
    j=0
    for k in range(0,train_num):
        tmax=0.0
        temp=0.0
        if(alpha[k]>0 and alpha[k]<C):
            Ej=error_cache[k]
            temp=np.fabs(Ei-Ej)
            if(temp>tmax):
                tmax=temp
                j=k
    if(j>=0 and  takeStep(i,j)): return 1
    else: return 0

#2. if step1 did not work, we start randomly in non_boundary samples
def examNonBound(i):
    rand = random.randint(0,train_num)
    for k in range(0,train_num):
        j=(k+rand)%train_num
        if((alpha[j]>eps and alpha[j]<C) and  i!=j): return 1
    return 0

#3. if step2 failed, we start randomly in the whole samples including the boundary

def examBound(i):
    rand = random.randint(0,train_num)
    for k in range(0,train_num):
        j=(k+rand)%train_num
        if(takeStep(i,j)): return 1
    return 0

def final_w():
    weight=np.zeros((1,4))[0]
    for i in range(0,4):
        for k in range(0,train_num):
            weight[i]+=alpha[k]*X[k][i]*Y[k]
    return weight

def supportVector():
    supVec={}
    count=0
    for i in range(0,train_num):
        if alpha[i] >0.0:
            supVec[count]=X[i]
            count+=1
    return supVec
'''
def SMO(filename,alpha,b,train_num,offset):

    fx=[]
    cout_err=0
    for i in range(0,train_num):
        sum=0.0
        for k in range(0,4):
            sum+=alpha[k]*Y[i]*K[k][i]
        fx.append(sum+b)
        if(fx[i]>0): fx[i]=1
        else: fx[i]=-1
        count_err=0
    for j in range(0,train_num):
        if(fx[j]!=Y[j]): count_err+=1
    w=final_w()
    supVec=supportVector()
# ToDo
    count_err
    err_rate=cout_err/train_num
    return [alpha,b,w,supVec,err_rate]
    '''

def predict(w,b,test_num,offset):
    err=0
    for i in range(test_num,test_num+offset):
        fy=np.dot(w,X[i])+b
        if  (fy >= 0 and Y[i] <0) or (fy < 0 and Y[i] > 0):
            err +=1
    return float(err)/test_num

if __name__ == '__main__':
    numChanged=0
    cout = 0
    examineAll=1 #outer loop flag
    get_xy('mystery.data',train_num,0)#Initialize data as X, Y
    xProduct()
    while(numChanged>0 or examineAll==1):
        numChanged=0
        cout+=1
        print "cout:", cout
        if(examineAll==1):#all multipliers
            for k in range(0,train_num):
                numChanged+=examineExample(k);
        else:# non_bound multipliers
            for k in range(0,train_num):
                if(alpha[k]!=0 and alpha[k]!=C):
                    numChanged+=examineExample(k)
                    print "alphak:",k,alpha[k]
        if (examineAll==1): examineAll=0
        elif numChanged>0: examineAll=1
        print "examineAll:", examineAll
    w=final_w()
    supVec=supportVector()

    print "alpha:", alpha
    print "b:", b
    print "w:", w
    for i in supVec:
        print "supVec:", supVec[i]
    #print "err_rate", model[4]
    err=predict(w,b,1000,0)
    print "err:",err
