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
dFileName = sys.argv[1]
lFileName = sys.argv[2]
d = getFeatureData(dFileName)
l = getLabelData(lFileName)

 ######Test for Local file(sample)
# d = getFeatureData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.data")#l
# l = getLabelData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.trainlabels")#dict

 ######Test for Local file(breast cancer)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.trainlabels.0")#dict


 ######Test for Local file(ionosphere)
# d = getFeatureData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.data")#l
# l = getLabelData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.trainlabels.0")#dict
#print(d)

######put bias values
bias_data=[] #all x includes missing label x 
for i in range(len(d)):
    bias_data.append([1]+d[i])
#print (bias_data)
r_train = [] #r, which datas with laebls. when label=0, ri=-1. elese ri=1 
test_label = []#when r missing ,which means without labels. 
train_data = [] #x, only includes trian data
for i in range(len(d)):
    if l.get(i) == 0:
        r_train.append(-1)
        train_data.append(bias_data[i])
    elif l.get(i) == 1:
        r_train.append(1)
        train_data.append(bias_data[i])
    else:
        test_label.append(i)

#print (r_train)
#print (test_label)
#print (train_data)

#####Initialization#######
eta = 0.001
theta = 0.000000001
w=[]
for j in range(len(train_data[0])):
    w.append(random.uniform(0.01,-0.01))
#print(w)
#w=[0.001]*len(train_data[0])
#print(w)
######Gradient descent iteration
#####compute output diff vector(1st)
d=[]
for i in range(len(train_data)):
    d.append(max(0,(1-r_train[i]*dotProduct(w,train_data[i]))))
#print(d)

####### set up for a identifty for storing data result for recognize label
Idl=[]
for i in range(len(train_data)):
    if r_train[i]*dotProduct(w,train_data[i])<1:
        Idl.append(1)
    else:
        Idl.append(0)
#print(Idl)


#####update w(1st)
for i in range(len(train_data)):
    for j in range(len(train_data[0])):
        w[j] = w[j]+ (eta*r_train[i]*Idl[i]*train_data[i][j])
        
#print(w)

####compute error(1st)
previous_error= sum(d)
#print(previous_error)

#####compute output diff vector(from 2nd to i)
for k in range(100000):
    d=[]
    for i in range(len(train_data)):
        d.append(max(0,(1-r_train[i]*dotProduct(w,train_data[i]))))
    error = sum(d)
    if abs(previous_error-error) <= theta:
        break

    ####### set up for a identifty for storing data result for recognize label
    Idl=[]
    for i in range(len(train_data)):
        if r_train[i]*dotProduct(w,train_data[i])<1:
            Idl.append(1)
        else:
            Idl.append(0)
#print(Idl)

    #####update w
    for i in range(len(train_data)):
        for j in range(len(train_data[0])):
            w[j] = w[j]+ (eta*r_train[i]*Idl[i]*train_data[i][j])
        

    ####compute previous error
    previous_error=sum(d)

#print(k)
    
print("w0=")
print(w[0])
print("")
print("w'=")
print(w[1:len(train_data)])
print("")
print("distance from origin =",abs(w[0]/(dotProduct(w[1:len(train_data)],w[1:len(train_data)]))**0.5))
#print(abs(w[0]/(dotProduct(w[1:len(train_data)],w[1:len(train_data)]))**0.5))
print("")


for i in (test_label):
    d1=dotProduct(bias_data[i],w)
#print(d1)
    if d1 >0:
        print(1,i)
    else:
        print(0,i)
