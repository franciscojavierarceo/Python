# execfile("/Users/franciscojavierarceo/MyPrograms/Python/KL_Dist.py")
import scipy
import random
execfile("/Users/franciscojavierarceo/MyPrograms/Python/My_Functions.py")

x = []
for i in range(1,101):
    x.append(random.uniform(0,100))

e = []
for i in range(1,101):
    e.append(random.uniform(0,1))

y = []
for i,idx in enumerate(x):
    y.append(x[i] + e[i])

kldist = scipy.stats.entropy
print kldist(x,y)
print kldist(y,x)
# mylift(y,x)
# gini(y,y)
# gini(y,x)
# normgini(y,x)
# histogram(y,20)
histogram(y,20)
denplot(y)
cdfplot(y)