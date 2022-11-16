# P13. Apriori Algorithm Implementation on Trasaction Dataset 

#Given Dataset
t1 = list([1,2,3,4])
t2 = list([1,2,4])
t3 = list([1,5,6])
t4 = list([1,4,5])
t5 = list([2,4,5])

print("========= 1 frequency =========")
# 1 frequent item set
#f1 = np.array([0,0,0,0,0,0,0,0])
f1 = list([0,0,0,0,0,0,0])
tt = 5
for i in range(6 + 1):
    f1[i] += t1.count(i)
    f1[i] += t2.count(i)
    f1[i] += t3.count(i)
    f1[i] += t4.count(i)
    f1[i] += t5.count(i)
    
print(f1)

# find support
for i in range(6 + 1):
    f1[i] = (f1[i] / tt) * 100;

print("f1 : ",f1)

# check support > 50 or not 
f1_res = list()
for i in range(1,6 + 1):
    if(f1[i] >= 50):
        f1_res.append(i)

print("f1_res : ",f1_res) 


if(len(f1_res) == 0):
    print("No further computation... after 1 frequency")
    exit

print("========= 2 frequency =========")
# 2 frequent item set
f2 = list([])
for i in range(len(f1_res)):
    for j in range(i+1,len(f1_res)):
        k=0
        if((t1.count(f1_res[i]) > 0) and (t1.count(f1_res[j]) > 0)):
            #print("t1 : ",f1_res[i],":",f1_res[j])
            k = k + 1
        if(t2.count(f1_res[i]) > 0 and t2.count(f1_res[j]) > 0):
            #print("t2 : ",f1_res[i],":",f1_res[j])
            k += 1
        if(t3.count(f1_res[i]) > 0 and t3.count(f1_res[j]) > 0):
            #print("t3 : ",f1_res[i],":",f1_res[j])
            k += 1
        if(t4.count(f1_res[i]) > 0 and t4.count(f1_res[j]) > 0):
            #print("t4 : ",f1_res[i],":",f1_res[j])
            k += 1
        if(t5.count(f1_res[i]) > 0 and t5.count(f1_res[j]) > 0):
            #print("t5 : ",f1_res[i],":",f1_res[j])
            k += 1
        f2.append(list([f1_res[i],f1_res[j],k]))


print(f2)

# find support
for i in range(len(f2)):
    f2[i][2] = (f2[i][2] / tt) * 100;

print("f2 : ",f2)

# check support > 50 or not 
f2_res = list()
f2_res2 = list([])
for i in range(len(f2)):
    if(f2[i][2] >= 50):
        f2_res.append(f2[i][0])
        f2_res.append(f2[i][1])
        f2_res2.append(f2[i])

print("f2_res : ",f2_res) 

if(len(f2_res) == 0):
    print("No further computation... after 2 frequency")
    exit
    
# confidence
fin = list([])
print("F2 res2 : ",f2_res2)
for f in f2_res2:
    # check confidence > 75 or not
    fin.append(list([f[0],f[1],(f[2]/f1[f[0]])*100]))
    fin.append(list([f[1],f[0],(f[2]/f1[f[1]])*100]))
    print("Confidence( ",f[0],"->",f[1]," )",f[2]/f1[f[0]])
    print("Confidence( ",f[1],"->",f[0]," )",f[2]/f1[f[1]])

print(fin)
final = list([])
for f in fin:
    if(f[2] >= 75):
        final.append(f)

print("Final : ",final)

print("\n========== Final Rules According to 2 frequency==============")
for i in fin:
    print(i[0],"->",i[1])
print()

print("========= 3 frequency =========")   
# 3 frequency item set
f2_res_old = f2_res
f2_res = set(f2_res)
f2_res = list(f2_res)
f3 = list([])
for i in range(len(f2_res)):
    for j in range(i+1,len(f2_res)):
        for l in range(j+1,len(f2_res)):
            k=0
            if(t1.count(f2_res[i]) > 0 and t1.count(f2_res[j]) > 0 and t1.count(f2_res[l]) > 0):
                #print("t1 : ",f2_res[i],":",f2_res[j],":",f2_res[l])
                k = k + 1
            if(t2.count(f2_res[i]) > 0 and t2.count(f2_res[j]) > 0 and t2.count(f2_res[l]) > 0):
                #print("t2 : ",f2_res[i],":",f2_res[j],":",f2_res[l])
                k += 1
            if(t3.count(f2_res[i]) > 0 and t3.count(f2_res[j]) > 0 and t3.count(f2_res[l]) > 0):
                #print("t3 : ",f2_res[i],":",f2_res[j],":",f2_res[l])
                k += 1
            if(t4.count(f2_res[i]) > 0 and t4.count(f2_res[j]) > 0 and t4.count(f2_res[l]) > 0):
                #print("t4 : ",f2_res[i],":",f2_res[j],":",f2_res[l])
                k += 1
            if(t5.count(f2_res[i]) > 0 and t5.count(f2_res[j]) > 0 and t5.count(f2_res[l]) > 0):
                #print("t5 : ",f2_res[i],":",f2_res[j],":",f2_res[l])
                k += 1
            f3.append(list([f2_res[i],f2_res[j],f2_res[l],k]))


print(f3)

# find support
for i in range(len(f3)):
    f3[i][3] = (f3[i][3] / tt) * 100;

print("f3 : ",f3)

# check support > 50 or not 
f3_res = list()
for i in range(len(f3)):
    if(f3[i][3] >= 50):
        f3_res.append(f3[i][0])
        f3_res.append(f3[i][1])
        f3_res.append(f3[i][2])

print("f3_res : ",f3_res)


if(len(f3_res) == 0):
    print("No further computation... after 3 frequency")
    
    exit


'''
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd
from apyori import apriori

#Importing the dataset  
dataset = pd.read_csv('Market_Basket_data1.csv')
print(dataset)
transactions=[]  
for i in range(0, 7501):  
    transactions.append([str(dataset.values[i,j])  for j in range(0,20)])
 
rules= apriori(transactions= transactions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=2)

results= list(rules)  
results

for item in results:  
    pair = item[0]   
    items = [x for x in pair]  
    print("Rule: " + items[0] + " -> " + items[1])  
  
    print("Support: " + str(item[1]))  
    print("Confidence: " + str(item[2][0][2]))  
    print("Lift: " + str(item[2][0][3]))  
    print("=====================================")
'''



