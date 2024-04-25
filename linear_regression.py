from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = [[1, 1], [1, 2], [2, 2], [2, 3]]  # Example features
y = [6, 8, 9, 11]  # Target variable

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating an instance of Linear Regression
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Displaying coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
