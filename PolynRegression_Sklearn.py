import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x_plot = np.linspace(0, 50, 1000)

x = np.array([0, 10, 20, 30, 40, 50], dtype=np.longdouble)
y = np.array([0, 102.6903, 105.4529, 81.21744, 55.6016, 35.6859])

# Column matrix
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.scatter(x, y, s=30, marker='x', label='data')
degreelst = [1, 2, 3, 4]

for degree in degreelst:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression(fit_intercept=False))
    model.fit(X, y)
    polyn = model.named_steps['linearregression']
    coeff = polyn.coef_
    print('Coefficients:', coeff)
    print('R^2:', model.score(X, y))
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label=f'degree {degree}')

plt.grid(alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
