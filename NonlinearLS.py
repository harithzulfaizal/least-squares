import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def func(x, a, b):
    return a*x/(b+np.exp(x))

def Jacobian(f, x, a, b):
    h = 1e-6
    pdv_a = (f(x, a+h, b)-f(x, a-h, b))/(2*h) # Finite difference: pdvF = (F(x+h)-F(x-h))/(2h)
    pdv_b = (f(x, a, b+h)-f(x, a, b-h))/(2*h)
    return np.column_stack([pdv_a, pdv_b])

def GaussNewton(f, x, y, a0, b0, iterations): # Gauss-Newton
    tol = 1e-12
    g_i = g = np.array([a0, b0])
    for itr in range(iterations):
        g = g_i
        J = Jacobian(f, x, g[0], g[1])
        r = y - f(x, g[0], g[1])
        g_i = g + np.linalg.inv(J.T@J)@J.T@r # \beta_{i+1} = \beta{i} + (J_i^T J_i)^{-1} J_i^T r_i
        if np.linalg.norm(g-g_i) < tol:
            break
    return g

def r(a, x, y):
    return a[0]*x/(a[1]+np.exp(x)) - y

def main():
    x = np.linspace(0, 10, 50)
    y = func(x, 6, 9) + np.random.normal(0, 0.02, x.shape) # + Noise

    a, b = GaussNewton(func, x, y, 2.5, 0.6, 50)
    print(a, b)
    y_hat = func(x, a, b)

    a_guess = np.array([2, 3])
    res = least_squares(r, a_guess, args=(x, y)) # Scipy's least squares module
    res_cauchy = least_squares(r, a_guess, loss='cauchy', f_scale=0.02, args=(x, y)) # Cauchy loss function
    print(res.x)
    y_hatscp = func(x, res.x[0], res.x[1])

    plt.plot(x, y, 'x')
    plt.plot(x, y_hat, label=r'Gauss-Newton')
    plt.plot(x, y_hatscp, label=r'least_squares')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
