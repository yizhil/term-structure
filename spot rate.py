# linear interpolation
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\joeyb\Desktop\mid-term project\par_yield.csv')
coupon=list(data.rate)
coupon=[x/100 for x in coupon]

def func(z):
    F=(coupon[n-1]/2+1)*100/(1+z/2)**n-100
    for i in range(n-1):
        F=F+coupon[n-1]/2*100/(1+coupon[i]/2)**(i+1);
    return F

spot_rate_linear=[coupon[0],coupon[1]]
for n in range(3,61):
    spot_rate_linear.append(float(fsolve(func,0.1)))

maturity=[x/2 for x in range(1,61)]
plt.plot(maturity,spot_rate_linear)
plt.title("Spot curve(linear interpolation)")
plt.xlabel("Maturity")
plt.ylabel("Spot rate")
plt.show()


#cubic interpolation
import numpy as np

y=[1.375/100,2.625/100,2.75/100,2.875/100,0.03,3.375/100]
t=[0.5,1,2,5,12,27]
t_mat1=np.array([[t[0]**3,t[0]**2,t[0],1],[t[1]**3,t[1]**2,t[1],1],[t[2]**3,t[2]**2,t[2],1],[t[3]**3,t[3]**2,t[3],1]])
y_mat1=np.array([[y[0]],[y[1]],[y[2]],[y[3]]])
para1=np.linalg.inv(t_mat1).dot(y_mat1)
para_1=[]
for i in range(4):
    para_1.append(float(para1[i]))
t_1=[x/2 for x in range(1,11)]

spot_rate_cubic=[]
for x in t_1:
    spot_rate_cubic.append(para_1[0]*x**3+para_1[1]*x**2+para_1[2]*x+para_1[3])

t_mat2=np.array([[t[2]**3,t[2]**2,t[2],1],[t[3]**3,t[3]**2,t[3],1],[t[4]**3,t[4]**2,t[4],1],[t[5]**3,t[5]**2,t[5],1]])
y_mat2=np.array([[y[2]],[y[3]],[y[4]],[y[5]]])
para2=np.linalg.inv(t_mat2).dot(y_mat2)
para_2=[]
for i in range(4):
    para_2.append(float(para2[i]))
t_2=[x/2 for x in range(11,61)]
for x in t_2:
    spot_rate_cubic.append(para_2[0]*x**3+para_2[1]*x**2+para_2[2]*x+para_2[3])

maturity=[x/2 for x in range(1,61)]
plt.plot(maturity,spot_rate_cubic)
plt.title("Spot curve(cubic interpolation)")
plt.xlabel("Maturity")
plt.ylabel("Spot rate")
plt.show()

#swap rate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\joeyb\Desktop\mid-term project\swap.csv')
swap=[x/100 for x in list(data.swap)]
def discount(df):
    F=(swap[n]*100/2+100)*df-100
    for i in range(n):
        F=F+swap[n]*100/2*discount_factor[i]
    return F

discount_factor=[]
for n in range(60):
    discount_factor.append(float(fsolve(discount,0.1)))
    
def swap_spot(s):
    return discount_factor[n]-1/(1+s/2)**(n+1)

spot_rate_swap=[]
for n in range(60):
    spot_rate_swap.append(float(fsolve(swap_spot,0.1)))

maturity=[x/2 for x in range(1,61)]
plt.plot(maturity,spot_rate_swap)
plt.title("Spot curve(swap rate)")
plt.xlabel("Maturity")
plt.ylabel("Spot rate")
plt.show()

#misprice
import math as m
def misprice(maturity,payment,coupon,spot_rate): 
    #annual payment 
    if payment==1: 
        price=(100+100*coupon)/(1+spot_rate[maturity*2-1])**maturity
        for i in range(m.ceil(maturity)-1):
            price+=100*coupon/(1+spot_rate[(maturity-i-1)*2-1])**(maturity-1-i)
    #semi-annual payment 
    else:
        price=(100+100*coupon/2)/(1+spot_rate[int(maturity*2-1)])**maturity
        for i in range(int(maturity*2-1)):
            print(price)
            price+=100*coupon/2/(1+spot_rate[int((maturity-0.5*i-0.5)*2-1)])**(maturity-0.5-0.5*i)
    return price

actual_price=[122.609375,99.71875,109.84375]
print('Actual bond price\t\t',actual_price)
#linear
bond_linear=[]
bond_linear.append(misprice(10,1,0.0525,spot_rate_linear))
bond_linear.append(misprice(10,1,0.02625,spot_rate_linear))
bond_linear.append(misprice(20,1,0.035,spot_rate_linear))
print('Linear interpolation bond price ',bond_linear)
#cubic
bond_cubic=[]
bond_cubic.append(misprice(10,1,0.0525,spot_rate_cubic))
bond_cubic.append(misprice(10,1,0.02625,spot_rate_cubic))
bond_cubic.append(misprice(20,1,0.035,spot_rate_cubic))
print('Cubic interpolation bond price  ',bond_cubic)
#swap rate
bond_swap=[]
bond_swap.append(misprice(10,1,0.0525,spot_rate_swap))
bond_swap.append(misprice(10,1,0.02625,spot_rate_swap))
bond_swap.append(misprice(20,1,0.035,spot_rate_swap))
print('Swap rate bond price\t\t',bond_swap)
