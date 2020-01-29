import sys
import random

####read data 
def getFeatureData(featureFile,bias=0):
    x=[]
    dFile = open(featureFile, 'r')
    i=0
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        if bias > 0:
            rVec.insert(0,bias)
        #print('row {} : {}'.format(i,rVec))
        x.append(rVec)
        i += 1
    dFile.close()
    return x
####read labels 

def getLabelData(labelFile,hyperPlaneClass=False):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        #print('label : {}'.format(lDict))
        if hyperPlaneClass and int(row[0]) <= 0:
            lDict[int(row[1])] = -1
        else:
            lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

######### Definition of the dot product

def dotProduct(u,v):
    sum = 0
    if len(u) != len(v):
        print("these two vectors are not equal")
    else:
        for i in range(len(u)):
            sum += u[i] * v[i]
        return sum

#u=[1,2,3]
#v=[1,3,3]
#print(dotProduct(u,v))

 ####### read the file            
#dFileName = sys.argv[1]
#lFileName = sys.argv[2]
#d = getFeatureData(dFileName)
#l = getLabelData(lFileName)

 ######Test for Local file(sample)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.trainlabels")#dict

 ######Test for Local file(breast cancer)
d = getFeatureData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.data")#l
l = getLabelData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.trainlabels.0")#dict


 ######Test for Local file(ionosphere)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.trainlabels.0")#dict
#print(d)

######put bias values
r_train = [] #r, which datas with laebls. when label=0, ri=-1. elese ri=1 
test_label = []#when r missing ,which means without labels. 
train_data = [] #x, only includes trian data
for i in range(len(d)):
    if l.get(i) == 0:
        r_train.append(-1)
        train_data.append(d[i])
    elif l.get(i) == 1:
        r_train.append(1)
        train_data.append(d[i])
    else:
        test_label.append(i)

#print (r_train)
#print (test_label)
#print (train_data)
        
##########Calculate the gini valume for each column
col_size = len(train_data[0])
row_size = len(train_data)

for j in range(col_size):
    gini = []
    for i in range(row_size):
        lsize = 0
        lp = 0
        rsize = 0
        rp =0
        for k in range(row_size):
            if train_data[k][j] <= train_data[i][j]:
                lsize += 1
                if r_train[k] == -1:
                    lp += 1
            elif train_data[k][j] > train_data[i][j]:
                rsize += 1
                if r_train[k] == -1:
                    rp += 1
        
        if lsize == 0 :
            gini.append((rsize/row_size)*(rp/rsize)*(1-rp/rsize))
        elif rsize == 0:
            gini.append((lsize/row_size)*(lp/lsize)*(1-lp/lsize))
        else:
            gini.append((lsize/row_size)*(lp/lsize)*(1-lp/lsize)+(rsize/row_size)*(rp/rsize)*(1-rp/rsize))

    index = gini.index(min(gini))
    print(j,train_data[index][j])
            
        
                        
             









