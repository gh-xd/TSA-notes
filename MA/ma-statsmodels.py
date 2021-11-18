from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt

# make dataset
data = [x + np.random.normal(0, 20) for x in range(1,100)]

# fit the model, set order = 1 equals to MA model
model_order1 = ARIMA(data, order=(0,0,1))
model_fit_order1 = model_order1.fit()

# just see the difference when order = 2
model_order2 = ARIMA(data, order=(0,0,2))
model_fit_order2 = model_order2.fit()

# just see the difference when order = 3
model_order3 = ARIMA(data, order=(0,0,1))
model_fit_order3 = model_order3.fit()


# make the prediction
y_hat1 = model_fit_order1.predict(len(data), len(data))
y_hat2 = model_fit_order2.predict(len(data), len(data))
y_hat3 = model_fit_order3.predict(len(data), len(data))


plt.plot(data)
plt.scatter(len(data)+1, y_hat1, color='red', marker='*')
plt.scatter(len(data)+1, y_hat2, color='blue', marker='o')
plt.scatter(len(data)+1, y_hat3, color='yellow', marker='.')
plt.text(len(data)-5, y_hat1+5, 'prediction: \n%.4f'%y_hat1, color='red')
plt.title("MA")
plt.show()