import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from TFANN import ANNR
from google.colab import files

files.upload()

stock_data = np.loadtxt('stockinfo.csv', delimiter=",", skiprows=1, usecols=(1, 4))
stock_data=scale(stock_data)

prices = stock_data[:, 1].reshape(-1, 1)
dates = stock_data[:, 0].reshape(-1, 1)

mpl.plot(dates[:, 0], prices[:, 0])
mpl.show()

# Neurons in each layer
input = 1
output = 1
hidden = 50

layers = [('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', output)]

# Construct the model and dictataQe params
mlpr = ANNR([input], layers, batchSize = 256, maxIter = 20000, tol = 0.2, reg = 1e-4, verbose = True)


# Specify hold out period
holdDays = 5
totalDays = len(dates)

# Fit the model to the data - "Learning"
mlpr.fit(dates[0:(totalDays-holdDays)], prices[0:(totalDays-holdDays)])


# Predict the stock price using the model
pricePredict = mlpr.predict(dates)

#Display the predicted reuslts agains the actual data
mpl.plot(dates, prices)
mpl.plot(dates, pricePredict, c='#5aa9ab')
mpl.show()

