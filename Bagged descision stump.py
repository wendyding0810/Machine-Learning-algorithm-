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
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.trainlabels.0")#dict


 ######Test for Local file(ionosphere)
d = getFeatureData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.data")#l
l = getLabelData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.trainlabels.0")#dict
#print(d)

 ######Test for Local file(climate)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\climate_simulation\climate.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\climate_simulation\climate.trainlabels.1")#dict
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
        

##############def function##########
def gini(x,r):
    col_size = len(x[0])
    row_size = len(x)
    output = {}
    gini_min_value = 100
    
    for j in range(col_size):
        vector = []
        for i in range(len(x)):
            if x[i][j] not in vector:
                vector.append(x[i][j])
        vector.sort()
        vector=[min(vector)-0.5]+vector
        for k in range(1,len(vector)-1):
            vector[k]=((vector[k]+vector[k+1])/2)
        vector[len(vector)-1]=max(vector)+0.5
        #print(vector)
        
        gini_inpurity = []
        for k in range(len(vector)):
            lsize = 0
            lp = 0
            rsize = 0
            rp =0
            for i in range(row_size):
                if x[i][j] < vector[k]:
                    lsize += 1
                    if r[i] == -1:
                        lp += 1
                elif x[i][j] > vector[k]:
                    rsize += 1
                    if r[i] == -1:
                        rp += 1
                        
            if lsize == 0 :
                gini_inpurity.append((rsize/row_size)*(rp/rsize)*(1-rp/rsize))
            elif rsize == 0:
                gini_inpurity.append((lsize/row_size)*(lp/lsize)*(1-lp/lsize))
            else:
                gini_inpurity.append((lsize/row_size)*(lp/lsize)*(1-lp/lsize)+(rsize/row_size)*(rp/rsize)*(1-rp/rsize))
                
                
        index = gini_inpurity.index(min(gini_inpurity))
        if min(gini_inpurity) < gini_min_value:
            output["col"]= j  #which column will give you minimal gini inpurity value
            output["s"] = vector[index]  #value of the minimal gini velue
            gini_min_value=min(gini_inpurity)
    return output


def majority(data, label, c, t):
    majority = [0,0]
    for i in range(len(data)):
        if data[i][int(c)] < t:
            if label[i] == -1:
                majority[0] += 1
            else:
                majority[1] += 1
    if majority[0] > majority[1]:
        return -1 # -1 is more than 1
    else:
        return 1
        
# print(gini(train_data,r_train))
# print(majority(train_data,r_train, gini(train_data,r_train)["col"], gini(train_data,r_train)["s"]))            


vote = {}
for i in test_label:
    vote[i]=0


L=100
for k in range(L):
    BS_data = []
    BS_label = []    
    for i in range (len(train_data)):
        index = random.randint(0,len(train_data)-1)
        BS_data.append(train_data[index])
        BS_label.append(r_train[index])
    
    split = gini(BS_data,BS_label)
    #print(split)
    col = split["col"]
    s = split["s"]
    maj = majority(train_data, r_train, col, s)
    for j in test_label:
        if d[j][col] < s:
            vote[j] += maj
        else:
            vote[j] -= maj
            
for i in test_label:
    if vote[i] < 0:
        l[i]=0
    else:
        l[i]=1
        
for i in test_label:
    print(i,l[i])
            
        

 
    




                        
             










