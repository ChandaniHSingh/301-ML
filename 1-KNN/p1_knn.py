from math import sqrt
 
#Given Dataset
film = list(['A','B','C','D','E','F','G','H'])
fight = list([4,1,5,2,3,6,2,5,3])
song = list([3,4,2,5,4,3,4,4,2])
movieClass = list(['A','R','A','R','R','A','R','A'])


# fight seen and song of I movie
x2 = 3 #fight
y2 = 2 #song


# calcute Euclidian Ditance from I
ev = list()

for i in range(8):
    ev.append(sqrt(((fight[i]-x2)**2) + ((song[i]-y2)**2)))


# sort movieClass and ev in ascending order
for i in range(8):
    for j in range(8):
        if(ev[i] < ev[j]):
           temp = ev[i]
           ev[i] = ev[j]
           ev[j] = temp
           temp = movieClass[i]
           movieClass[i] = movieClass[j]
           movieClass[j] = temp

# Calculate k 
k = list()
temp = ev[0]
cnt = 1
for i in range(8):
    if(temp < ev[i]):
        cnt += 1
        temp = ev[i]
    k.append(cnt)

# printing k , ev, movieClass           
for i in range(8):
    print(k[i],"\t",ev[i],"\t",movieClass[i])


# create num from k unique value of k
num = list(k)
num = set(num)
num = list(num)

print("\n\nfor I movie Prediction \n\n")
# logic for Predict MovieClass of I 
for i in num:
    a=0
    r=0
    for j in range(8):
        if(k[j]>i):
            break
        else:
            if(movieClass[j] == 'A'):
                a += 1
            elif(movieClass[j] == 'R'):
                r += 1
    if(a==r):
        print("K = ",i," : movieClass = Random")
    elif(a>r):
        print("K = ",i," : movieClass = Action")
    else:
        print("K = ",i," : movieClass = Romantic")


