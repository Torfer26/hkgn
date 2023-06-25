# # Data Manipulation libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib
# import numpy as np

# # Load the dataset
# df = pd.read_csv('idealistadatos-numerico.csv', sep=';')  # Load the dataset

# df_x = df[['rooms', 'area_m2']]
# df_y = df[['prices']]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(df_x)

# df_x_scaled = scaler.transform(df_x)
# df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)
# X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, df_y, test_size = 0.33, random_state = 5)

# mlp = MLPRegressor(hidden_layer_sizes=(60), max_iter=1000)
# mlp.fit(X_train, Y_train)
# Y_predict = mlp.predict(X_test)

# #Saving the machine learning model to a file
# joblib.dump(mlp, "model/rf_model.pkl")

# Data Manipulation libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv('idealistadatos-numerico.csv', sep=';')

# Extract input features and target variable
df_x = df[['rooms', 'area_m2']]
df_y = df['prices']

# Scale input features
scaler = StandardScaler()
df_x_scaled = scaler.fit_transform(df_x)
df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, df_y, test_size=0.33, random_state=5)

# Train the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(60), max_iter=10000)
mlp.fit(X_train, Y_train)

# Make predictions on the test set
Y_predict = mlp.predict(X_test)

# Save the trained model to a file
joblib.dump(mlp, "model/mlp_model.pkl")




