import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.integrate import quad

func = lambda x: np.sin(np.pi*x) # Original function to be approximated
orth = [lambda x: 1, lambda x: x, lambda x: x**2-(1/3), lambda x: x**3-(3/5)*x] # Test

def c(f, phi2, a, b):
    num = quad(f, a, b) # Integral of f, limits a to b
    dnm = quad(phi2, a, b)
    return num[0]/dnm[0] # c_i = \frac{\langle f,\phi_i \rangle}{\langle \phi_i,\phi_i \rangle}

def y(order, x, coeff):
    polyn = 0
    for i in range(order + 1):
        polyn += coeff[i] * legendre(i)(x) # \hat{f}(x) = \sum_{i=1}^n c_i \phi_i(x)
    return polyn

def main():
    order = 3
    a, b = -1, 1
    coeff = []
    xlin = np.linspace(a, b, 1000)

    for i in range(order+1):
        phi = legendre(i) # Orthogonal function -- Legendre polynomials

        f = lambda x: func(x)*phi(x) # Combining 2 functions
        phi2 = lambda x: phi(x)*phi(x)

        coeff.append(c(f, phi2, a, b))

    print(coeff)
    plt.plot(xlin, func(xlin), label=r'$\sin\,{\pi x}$')
    plt.plot(xlin, y(order,  xlin, coeff), '--', label=r'Approximation')
    plt.legend()
    plt.title('Least squares function approximation of $\sin\,{\pi x}$')
    plt.show()

if __name__ == '__main__':
    main()
