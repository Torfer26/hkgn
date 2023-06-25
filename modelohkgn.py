
# Importamos las librerías que vamos a necesitar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris

def ia(name,rooms,area_m2,prices,lift):
    
    data = pd.read_csv(r'C:\Users\Ferran\OneDrive - es.logicalis.com\Escritorio\hkgn\hkgn pf\datos_precios_casas.csv', sep=';')

    data_f = data[['rooms', 'area_m2', 'prices', 'compra', 'lift','eurometro']]

    X = data_f.drop('compra', axis=1)
    y = data_f['compra']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=0)

    # Instancia el escalador 
    scaler = MinMaxScaler()

    # Escala los datos
    X_train_scaled = scaler.fit_transform(X_train)

    # Recrea el dataset de entrenamiento con las columnas estandarizadas
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_train.head()

    # Escala el conjunto de prueba
    X_test_scaled = scaler.transform(X_test)

    # Recrea el dataset de prueba con las columnas estandarizadas
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_test.head()

    # Instancia el clasificador
    lr = LogisticRegression(max_iter=10000,random_state=42)

    # Entrena el clasificador
    lr.fit(X_train, y_train)

    # Realiza las predicciones
    y_pred_lr = lr.predict(X_test)

    new_data = {
        "name": [name],
        "rooms": [rooms],
        "area_m2": [area_m2],
        "prices": [prices],
        "lift": [lift],
        "eurometro": [round(int(prices) / int(area_m2))]
    }

    # Crea un DataFrame con los nuevos datos
    new_df = pd.DataFrame(new_data)

    # Establece 'name' como índice
    new_df.set_index('name', inplace=True)
    new_df

    # Escala los datos
    new_df_scaled = scaler.transform(new_df)

    # Recrea el dataset de prueba con las columnas estandarizadas
    new_df = pd.DataFrame(new_df_scaled, 
                        columns=new_df.columns, 
                        index=new_df.index)
    new_df.head()

    leyenda = {1: "Buena compra",
            0: "Mala compra"}

    # Haz las predicciones
    y_new = lr.predict(new_df)

    # Imprime los resultados
    #resultado = []
    #for i in range(len(new_df)):
    #resultado = leyenda[y_new[0]]
    #resultado = f"{name} es una {leyenda[y_new[0]]}."
    resultado = leyenda[y_new[0]]

    return resultado

# print(ia('hkgn', int('2'), int('200'), int('150000'), int('1')))

