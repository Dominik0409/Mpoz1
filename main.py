import numpy as np
import math as m

def test_lokalny(v,a,atas,s,u_kryt):
    Qv = np.identity(v.size) - (a.dot(atas)).dot(a.T)
    for i in range(v.size):
        u = (abs(v[i]))/(s*m.sqrt(Qv[i,i]))
        if u <= u_kryt:
            print(u,'OK')
        else:
            print(u,'-')

def sigma0(v,atas):
    sigma = m.sqrt((v.T.dot(v))/(12))
    sigmax = np.zeros((12,1))
    for i in range(12):
        sigmax[i] = sigma * m.sqrt(abs(atas[i,i]))
    return sigmax

sigma0_ap = 0.08
chi2= 21.026

f = open("dane1.txt")
data = []
for line in f:
    data.append([float(i) for i in str.split(line)])
f.close()

for i in data:
    if i[1] > 100: i[1] -= 93
    if i[2] > 100: i[2] -= 93

#wpisanie danych do macierzy
 
a = np.zeros((len(data),12))
l1 = np.empty((len(data),1))
l2 = np.empty((len(data),1))
l3 = np.empty((len(data),1))
p = np.empty((len(data),1))
s = np.zeros((12,1))
s[0] = 1
s1 = np.zeros((13,1))
s1[0] = 1

for i in range(len(data)):
    a[i][int(data[i][1])-1] = -1
    a[i][int(data[i][2])-1] = 1
    l1[i][0] = data[i][3]
    l2[i][0] = data[i][5]
    l3[i][0] = data[i][5] - data[i][3]
    p[i][0] = m.sqrt(1/data[i][4])

print(a)
print(l1)
print(p)

#macierze razy macierz wag

for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] *= p[i][0]
    l1[i][0] *= p[i][0]
    l2[i][0] *= p[i][0]
    l3[i][0] *= p[i][0]

print(a)

c1 = (a.T).dot(a)
c2_1 = (a.T).dot(l1)
c2_2 = (a.T).dot(l2)
c2_3 = (a.T).dot(l3)

c1 = np.append(c1, s, 1)
c1 = np.append(c1, s1.T, 0)

c1 = np.linalg.inv(c1)

print("c1 shape: ",c1.shape)
print(c1[0:-1,0:-1].shape)

X_1 = c1[0:-1,0:-1].dot(c2_1)
X_2 = c1[0:-1,0:-1].dot(c2_2)
X_3 = c1[0:-1,0:-1].dot(c2_3)

V_1 = a.dot(X_1) - l1
V_2 = a.dot(X_2) - l2
V_3 = a.dot(X_3) - l3

sigmap_0_1 = m.sqrt((V_1.T.dot(V_1))/(12))/sigma0_ap
sigmap_0_2 = m.sqrt((V_2.T.dot(V_2))/(12))/sigma0_ap
sigmap_0_3 = m.sqrt((V_3.T.dot(V_3))/(12))/sigma0_ap

sigma0_kryt = m.sqrt(chi2/12)

print(sigmap_0_1)
print(sigmap_0_2) 
print(sigmap_0_3) 
print(sigma0_kryt) 

test_lokalny(V_1,a,c1[0:-1,0:-1],sigma0_ap,2.5)

print(X_3)

print(c1[0:-1,0:-1])

sigmaV_3 = sigma0(V_3,c1[0:-1,0:-1])
print(sigmaV_3)