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
