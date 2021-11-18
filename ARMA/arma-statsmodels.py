from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt

# make dataset
data = [x + np.random.normal(0, 20) for x in range(1,100)]

# fit the model, set order = 1 equals to MA model
model_order = ARIMA(data, order=(2,0,1))
model_fit = model_order.fit()

# make the prediction
y_hat1 = model_fit.predict(len(data), len(data))

plt.plot(data)
plt.scatter(len(data)+1, y_hat1, color='red', marker='*')
plt.text(len(data)-5, y_hat1+5, 'prediction: \n%.4f'%y_hat1, color='red')
plt.title("MA")
plt.show()