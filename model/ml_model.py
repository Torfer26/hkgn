# Data Manipulation libraries
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv('idealistadatos.csv', sep=';')  # Load the dataset

column_names = df.columns
print(column_names)

df_x = df[['district', 'neighborhood', 'condition', 'type', 'rooms', 'area_m2', 'lift', 'views', 'floor']]
df_y = df[['prices']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_x)

df_x_scaled = scaler.transform(df_x)
df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)
X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, df_y, test_size = 0.33, random_state = 5)

mlp = MLPRegressor(hidden_layer_sizes=(60), max_iter=1000)
mlp.fit(X_train, Y_train)
Y_predict = mlp.predict(X_test)

#Saving the machine learning model to a file
joblib.dump(mlp, "model/rf_model.pkl")