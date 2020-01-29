import sys

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

###########mean and std
    
def mean_function(x):
    
    if type(x[0]) != list: 
        w=0
        for i in range(len(x)):
            w += x[i]/len(x)
    else:
        w = []
        for j in range(len(x[0])):
            d=0
            for i in range(len(x)):
                d += x[i][j]/len(x)
            w.append(d)
        return w


def std_function(x):
    meanx=mean_function(x)
    if type(x[0]) != list:
        w=0
        for i in range(len(x)):
            w += (x[i]-meanx)**2/len(x)
        return w**(1/2)
    else:
        w = []
        for j in range(len(x[0])):
            d=0
            for i in range(len(x)):
                d += (x[i][j]-meanx[j])**2/len(x)
            w.append(d**(1/2))
        return w
            
 ####### read the file            
dFileName = sys.argv[1]
lFileName = sys.argv[2]
d = getFeatureData(dFileName)#l
l = getLabelData(lFileName)#dict


#####Creating a matrix for training and test data
####Train_data
Train_c0 =[]
Train_c1 =[]
Test_data = []
Test_label = []
for i in range(len(d)):
    if l.get(i) == 0:
        Train_c0.append(d[i])
    elif l.get(i) == 1:
        Train_c1.append(d[i])
    else:
        Test_data.append(d[i])
        Test_label.append(i)
#print (Test_label)
#for i in range(len(Test_data)):
#    print (Test_data[i])

###initializtion-calculate mean and sd for the class
m0= mean_function(Train_c0)
m1= mean_function(Train_c1)
s0= std_function(Train_c0)
s1= std_function(Train_c1)

####cacluating the output 
#####naive bayes algorithm 
for i in (Test_label):
    nb_0 =0
    nb_1 =0
    for j in range(len(d[0])):
        nb_0 += ((d[i][j]-m0[j])/s0[j])**2
        nb_1 += ((d[i][j]-m1[j])/s1[j])**2
    if nb_0 < nb_1 :
        l[i]=0
    else:
        l[i]=1
    

#####print out the result
for i in (Test_label):
    print(l[i],i)
    
    





