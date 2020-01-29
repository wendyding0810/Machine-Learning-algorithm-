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

############ norm function of compute distance formula
############ this function is the absolue value for computing the distance formula
def Norm_Vector(u):
    sum=0
    for i in range(len(u)):
        sum += u[i]**2
    return sum**(1/2)


############# Definition of the vector minus
############# this function use for compute the distance(ref: material clstering1 p.11)
def Vector_Minus(u,v):
    if len(u) != len(v):
        print ("error u != v")
    else:
        vm=[]
        for i in range(len(u)):
            vm.append(u[i]-v[i])
        return vm

#########choosing cluster 
def c_c(x,c,k):
    output=[]
    for i in range(len(x)):
        if c[i]==k:
            output.append(x[i])
    return(output)

##########mean function
##### this is a def function for computing mean of each cluster  
def column_mean(x,y):
    data =len(x)
    if len(x) == 0:
        output=[]
        for j in range(y):
            output.append(0)
        return(output)    
    elif type(x[0]) != list:
        return(x)
    else:
        column_size=len(x[0])
        output=[]
        for j in range(column_size):
            column_sum=0
            for i in range(data):
                column_sum += x[i][j]/data
            output.append(column_sum)
        return(output)



####### read the file            
dFileName = sys.argv[1]
d = getFeatureData(dFileName)


 ######Test for Local file(sample)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\HW03\testSVM.trainlabels")#dict

 ######Test for Local file(breast cancer)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\breast_cancer\breast_cancer.trainlabels.0")#dict


 ######Test for Local file(ionosphere)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\ionosphere\ionosphere.trainlabels.0")#dict
#print(d)

 ######Test for Local file(climate)
#d = getFeatureData(r"C:\Users\wendy\OneDrive\data\climate_simulation\climate.data")#l
#l = getLabelData(r"C:\Users\wendy\OneDrive\data\climate_simulation\climate.trainlabels.1")#dict
#print(d)

######definition of how many cluster that we need to seperate  
#cluster_size =3
cluster_size = int(sys.argv[2])


######read the file

column_size=len(d[0])
data =len(d)

#######Initialization for random selection from the data
cluster = []
for i in range(data):
    cluster.append(random.randint(0,cluster_size-1))

###########initialization of m (ref: cluster1 p.11, step 2)
#########this step is computing the mean of each cluster.
#####use the def function that set up before to calculate mean velue. 

clsuter_mean=[]
for k in range(cluster_size):
    clsuter_mean.append(column_mean(c_c(d,cluster,k),column_size))

########### initialization of obj (ref: cluster1 p.11, step 3)
##### this step is calculating the distance from datapoints to mean of each cluster 
obj=0
for k in range(cluster_size):
    sub_data=c_c(d,cluster,k)
    sub_size=len(sub_data)
    for i in range(sub_size):
        obj += Norm_Vector(Vector_Minus(sub_data[i],clsuter_mean[k]))**2


############ Recompute cluster (cluster1 p.11 step 5)
        
for i in range(data):
    recompute=[]
    for k in range(cluster_size):
        recompute.append(Norm_Vector(Vector_Minus(d[i],clsuter_mean[k])))
    cluster[i]=recompute.index(min(recompute))


########## repeat the previous step of mean the distance (cluster1 p.11 step 5)
######### this step is computing mean and distance fot remaining data

for h in range(10000000):
    clsuter_mean=[]
    for k in range(cluster_size):
        clsuter_mean.append(column_mean(c_c(d,cluster,k),column_size))
    
    new_obj = 0
    for k in range(cluster_size):
        sub_data=c_c(d,cluster,k)
        sub_size=len(sub_data)
        if sub_size>0:
            for i in range(sub_size):
                new_obj += Norm_Vector(Vector_Minus(sub_data[i],clsuter_mean[k]))**2
    
    if abs(new_obj-obj) == 0 :
        break
    obj = new_obj
    for i in range(data):
        recompute=[]
        for k in range(cluster_size):
            recompute.append(Norm_Vector(Vector_Minus(d[i],clsuter_mean[k])))
        cluster[i]=recompute.index(min(recompute))
        

######## sort the cluster let the lager cluster abtain smaller index
cluster_list=[]
for k in range(cluster_size):
    cluster_list.append([])

for k in range(cluster_size):
    for i in range(data):
        if cluster[i]==k:
            cluster_list[k].append(i)
cluster_list.sort(key=len,reverse=True)

for k in range(cluster_size):
    for i in cluster_list[k]:
        cluster[i]=k

for i in range(data):
    print(cluster[i],i)

#for k in range(cluster_size):
#   print("The cluster",k," have ",cluster.count(k)," points") 







