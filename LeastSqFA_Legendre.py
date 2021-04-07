import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.integrate import quad

func = lambda x: np.sin(np.pi*x)
orth = [lambda x: 1, lambda x: x, lambda x: x**2-(1/3), lambda x: x**3-(3/5)*x] # Test

def c(f, phi2, a, b):
    num = quad(f, a, b)
    dnm = quad(phi2, a, b)
    return num[0]/dnm[0]

def y(order, x, coeff):
    polyn = 0
    for i in range(order + 1):
        polyn += coeff[i] * legendre(i)(x)
    return polyn

order = 3
a, b = -1, 1
coeff = []
xlin = np.linspace(a, b, 1000)

for i in range(order+1):
    phi = legendre(i)

    f = lambda x: func(x)*phi(x)
    phi2 = lambda x: phi(x)*phi(x)

    coeff.append(c(f, phi2, a, b))

print(coeff)
plt.plot(xlin, func(xlin))
plt.plot(xlin, y(order,  xlin, coeff), '--')
plt.show()
