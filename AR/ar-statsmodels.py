from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import matplotlib.pyplot as plt

# make dataset
data = [x + np.random.normal(0, 20) for x in range(1,100)]

# fit the model
model = AutoReg(data, lags=1)
model_fit = model.fit()

# make the prediction
y_hat = model_fit.predict(len(data), len(data))

plt.plot(data)
plt.scatter(len(data)+1, y_hat, color='red', marker='*')
plt.text(len(data)+1, y_hat+5, 'prediction: \n%.4f'%y_hat, color='red')
plt.title("AR")
plt.show()