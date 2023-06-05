from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Electric_Production.csv")
Y = data["IPG2211A2N"].to_numpy()
X = np.arange(len(Y))

# This should only be done for the training data
Y_mean = Y.mean()
Y_std = Y.std()
Y = (Y - Y_mean) / Y_std



# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=False)

kernel = RBF() * (ExpSineSquared() + ExpSineSquared() + ExpSineSquared())
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, normalize_y=True, alpha=0.1,)
gp.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
predictions,cov = gp.predict(X_test.reshape(-1,1), return_cov=True)
cov = np.diag(cov)

predictions = predictions * Y_std + Y_mean
y_test = y_test * Y_std + Y_mean
y_train = y_train * Y_std + Y_mean
cov = cov * Y_std**2

mae = np.mean(np.abs(predictions - y_test))
mae_percent = np.mean(np.abs(predictions - y_test) / y_test)
r2 = 1 - np.sum((predictions - y_test)**2) / np.sum((y_test - y_test.mean())**2)

print("MAE: ", mae)
print("MAE%: ", mae_percent)
print("R2: ", r2)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, label="Train data")
ax.set_title("Electricity production")
ax.set_xlabel("Time")
ax.set_ylabel("Electricity production")
ax.grid(True)
# Back ground to light grey
ax.set_facecolor((0.95,0.95,0.95))
ax.legend()

fig, ax = plt.subplots()
ax.plot(X_train, y_train, label="Train data")
ax.plot(X_test, predictions, label="Predictions")
ax.fill_between(X_test.ravel(), predictions.ravel() - 2*np.sqrt(cov), predictions.ravel() + 2*np.sqrt(cov), alpha=0.2, color="red", label="95% confidence interval")
ax.set_title("Electricity production")
ax.set_xlabel("Time")
ax.set_ylabel("Electricity production")
ax.grid(True)
# Back ground to light grey
ax.set_facecolor((0.95,0.95,0.95))
ax.legend()


fig, ax = plt.subplots()
ax.plot(X_train, y_train, label="Train data")
ax.plot(X_test, predictions, label="Predictions")
ax.fill_between(X_test.ravel(), predictions.ravel() - 2*np.sqrt(cov), predictions.ravel() + 2*np.sqrt(cov), alpha=0.2, color="red", label="95% confidence interval")
ax.plot(X_test, y_test, label="Test data")
ax.set_title("Electricity production")
ax.set_xlabel("Time")
ax.set_ylabel("Electricity production")
ax.grid(True)
# Back ground to light grey
ax.set_facecolor((0.95,0.95,0.95))
ax.legend()

# Plot only predictions and test data
fig, ax = plt.subplots()
ax.plot(X_test, y_test, label="Test data")
ax.plot(X_test, predictions, label="Predictions")
ax.fill_between(X_test.ravel(), predictions.ravel() - 2*np.sqrt(cov), predictions.ravel() + 2*np.sqrt(cov), alpha=0.2, color="red", label="95% confidence interval")
ax.set_title("Electricity production")
ax.set_xlabel("Time")
ax.set_ylabel("Electricity production")
ax.grid(True)
# Back ground to light grey
ax.set_facecolor((0.95,0.95,0.95))
ax.legend()





plt.show()

