import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import lstsq
from scipy.optimize import curve_fit

def func(order, x, *a):
    polyn = 0
    i = 0
    for p in range(order + 1):
        polyn += a[i] * x ** p
        i += 1
    return polyn

def get_gmatrices(order, x, y, n):
    z = order + 1
    A = np.empty(shape=(z, z))
    for i in range(z):
        itr = 0
        for j in range(z):
            if i == 0 and j == 0:
                A[i, j] = n
                itr += 1
            else:
                A[i, j] = sum(x ** (i + itr))  # Generalized matrix form for A
                itr += 1

    b = np.empty(shape=(z, 1))
    b[0, 0] = sum(y)
    for k in range(1, z):
        b[k, 0] = sum(y * (x ** k))

    return A, b

def get_lamatrices(order, x, y, n):
    power = list(range(order + 1))
    M = x[:, np.newaxis] ** power  # [1 x x^2 x^3 ...] form
    k = y.reshape(n, 1)

    return M, k

def get_error(y, fh):
    return np.sum((y - fh) ** 2)

def fcv(x, *s):
    return s[0] + s[1]*x + s[2]*x**2 # For curve_fit, can be any function

def main():
    df = pd.read_csv('annualglobaltemp.csv')

    xdata = df[['Year']].to_numpy().reshape(-1).T  # Down one dimension, transpose
    ydata = df[['Mean']].to_numpy().reshape(-1).T

    n = xdata.size
    x, y = xdata.astype('longdouble'), ydata
    xlin = np.linspace(x[-1], x[0], 1500)
    order = 3
    
    """ # Test dataset
    x = np.array([0, 10, 20, 30, 40, 50], dtype=np.longdouble)
    xlin = np.linspace(min(x), max(x), 1000)
    y = np.array([0, 102.6903, 105.4529, 81.21744, 55.6016, 35.6859])
    n = len(x)
    order = 2 """

    A, b = get_gmatrices(order, x, y, n)
    M, k = get_lamatrices(order, x, y, n)

    #np.linalg.solve can only use square matrix for A/ LU decomposition
    d = np.linalg.solve(A, b)
    d_err = get_error(y, func(order, x, *d.flatten()))
    print(d.flatten(), 'linalg.solve')
    print('Residual:', d_err)

    # For number of rows > columns (Not a square matrix)
    Mx = np.matmul(M.T, M)
    kx = np.matmul(M.T, k)
    c = np.linalg.solve(Mx, kx)
    print(c.flatten())
    print('Residual:', get_error(y, func(order, x, *c.flatten())))

    z, res, rank, s = lstsq(M, k) # SVD decomposition
    print(z.flatten())
    z_err = get_error(y, func(order, x, *z.flatten()))
    print('Residual:', z_err, res)

    p, _ = curve_fit(fcv, x, y, p0=[1,1,1])
    y_cv = fcv(x, *p)
    print(p, 'for curve_fit')


    plt.plot(x, y, 'x', label='Data')
    plt.plot(xlin, func(order, xlin, *d.flatten()), label=r'np.solve')
    plt.plot(xlin, func(order, xlin, *z.flatten()), label=r'scipy.lstsq')
    plt.plot(xlin, func(order, xlin, *c.flatten()), '--', label=r'np.solve $A^{T}$')
    plt.plot(xlin, fcv(xlin, *p), label=r'curve_fit')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Mean temperature anomalies ($^{\circ}$C)')
    plt.title('Global annual mean temperature anomalies')
    plt.show()


if __name__ == '__main__':
    main()
