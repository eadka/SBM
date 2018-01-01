# SIMILARITY BASED MODELLING TECHNIQUE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def SBM_eval(protArray,currVec):
    a_mat = a_val(protArray,currVec)
    g_vec = g_mat_AAKR(protArray.shape)
    w_mat = wt_mat(g_vec,a_mat)
    x_exp = ExpVal_calc(protArray,w_mat)
    return x_exp


# Simlarity Based Modelling Main Function 
def SBM_fun(path,dataFile):
    path = path + '/' + dataFile
    datain = pd.read_csv(path)
    TrainArray = datain
    del TrainArray['DATETIME']
    TrainArray = Trans_Arr(TrainArray)
    return TrainArray


def Trans_Arr(protArray):
    protArray = np.transpose(protArray)
    return protArray
    
    
def Sim_Fun(Array1,Array2):
    dist_arr = []
    dist_arr = np.sqrt((Array1-Array2)**2)
    dist_arr = 1/(1+dist_arr)
    return dist_arr
    
def a_val(protArray,currVec):
    an = pd.DataFrame(np.zeros(protArray.shape[0]))
    for i in range (0,protArray.shape[1]):
        buf = pd.DataFrame(np.zeros(protArray.shape[0]))
        buf = Sim_Fun(protArray[i],currVec)
        if i==0:
            an = pd.DataFrame(buf)
        else:
            an = np.concatenate((an,pd.DataFrame(buf)),axis=1)
    return an

def g_mat(protArray):
    gval = pd.DataFrame(np.zeros(protArray.shape[0]))
    #trans_mat = np.transpose(protArray)
    for i in range (0,protArray.shape[1]):
        for j in range (0,protArray.shape[1]):
            buf = pd.DataFrame(np.zeros(protArray.shape[0]))
            buf = Sim_Fun(protArray[i],protArray[j])
            if (i==0) & (j==0):
                gval = pd.DataFrame(buf)
            else:
                gval = np.concatenate((gval,pd.DataFrame(buf)),axis=1)
    return gval 
    
def g_mat_med(protArray):
    gval = pd.DataFrame(np.zeros(protArray.shape[0]))
    medval = pd.DataFrame(np.zeros(protArray.shape[0]))
    buf_val = protArray
    for j in range(0,protArray.shape[0]):
        medval.ix[j] = np.median(protArray.ix[j])
    for i in range (0,protArray.shape[1]):      
            buf = pd.DataFrame(np.zeros(protArray.shape[0]))
            buf = Sim_Fun(protArray[i],medval)
            if (i==0):
                gval = pd.DataFrame(buf[0])
            else:
                gval = np.concatenate((gval,pd.DataFrame(buf[0])),axis=1)
    return gval     
    
def g_mat_AAKR(dim):
    return np.ones([dim[0],dim[1]])

def wt_mat(ident_mat,a_val):
    return np.multiply(ident_mat,a_val)

def norm_one(w_mat):
    sum_val = 0
    for i in range(0,w_mat.shape[0]):
        for j in range(0,w_mat.shape[1]):
            sum_val += w_mat[i][j]**2
    return np.sqrt(sum_val)

def ExpVal_calc(protArray,weightVal):
    x_exp = []
    w_mat = []
    w_norm = norm_one(weightVal)
    w_mat = np.divide(weightVal,w_norm)
    x_exp = np.multiply(protArray,w_mat)
    x_exp = sum_vec(x_exp)
    return x_exp

def sum_vec(x_ex):
    sum_vec = np.zeros((x_ex.shape[0],1))
    for i in range(0,x_ex.shape[0]):
        sum_vec[i]=0
        for j in range(0,x_ex.shape[1]):
            sum_vec[i] += x_ex[j][i]
    return sum_vec

# Returns the absolute value of the residual
def res_calc(Training_val,exp_val):
    return (Training_val - exp_val)
    

path = '/home/neo/Desktop/aditi/similaritybasedmodelling'
dataFile = 'Tr.csv'
Training_Data = SBM_fun(path,dataFile)
residual_Val = Training_Data[1]
residual_Val[:] = 0

for i in range(0,Training_Data.shape[1]):
    x_exp = SBM_eval(Training_Data,Training_Data[i])
    residual_Val = res_calc(pd.DataFrame(Training_Data[i]),x_exp)
    # print x_exp
    plt.xlim([0,10])
    plt.ylim([-10,150])
    plt.scatter((i+1),Training_Data[i][2],color='blue',marker='o',label='Actual') #index 2 is best
    plt.scatter((i+1),x_exp[2],color='green',marker='v',label='Expected')
    plt.scatter((i+1),residual_Val.ix[2],color='red',marker='*',label='Residual')
    plt.tight_layout()
    plt.ylabel('Speed')
    plt.xlabel('Index')
    plt.legend(('Actual','Expected/ Estimate','Residual'))
    plt.show()
    