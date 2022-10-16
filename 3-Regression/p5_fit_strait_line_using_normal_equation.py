# find optimum value of theta0 (T0) and theta1 (T1) and fit strait-line using normal-equation

import numpy as np

design = "="*100

Experience = [2,3,5,13,8,16,11,1,9]
Salary = [15,28,42,64,50,90,58,8,54]

N = 9

X = np.array(Experience)
Y = np.array(Salary)

avgX = np.mean(X)
avgY = np.mean(Y)


print(design)

print("Mean of X : ",avgX)
print("Mean of Y : ",avgY)

X_dis = np.zeros(N)
Y_dis = np.zeros(N)

for i in range(N):
    X_dis[i] = ( X[i] - avgX )
    Y_dis[i] = ( Y[i] - avgY )



print(design)

print("X_dis : ",X_dis)
print("Y_dis : ",Y_dis)


X_dis_2 = np.power(X_dis,2)
XY_dis = np.multiply(X_dis,Y_dis)


print(design)

print("X_dis_2 : ",X_dis_2)
print("XY_dis : ",XY_dis)


X_dis_2_sum = 0
XY_dis_sum = 0

for i in X_dis_2:
    X_dis_2_sum += i

for i in XY_dis:
    XY_dis_sum += i


print(design)

print("X_dis_2_sum : ",X_dis_2_sum)
print("XY_dis_sum : ",XY_dis_sum)


T1 = XY_dis_sum / X_dis_2_sum

T0 = avgY - ( T1 * avgX )


print(design)

print("T1 : ",T1)
print("T0 : ",T0)


print("strait-line : ",T0,"+",T1,"x")
