<<<<<<< HEAD
from flask import Flask, render_template, jsonify, request
import os
import pandas as pd

app = Flask(__name__)
    
=======

from flask import Flask, render_template, request
import os
import pandas as pd
from modelohkgn import ia

app = Flask(__name__)

>>>>>>> 9c57d534ee4a2fe47353fe1379d505e514141a64
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    name = request.form['name']
    rooms = request.form['rooms']
    area_m2 = request.form['area_m2']
    prices = request.form['prices']
    lift = request.form['lift']
<<<<<<< HEAD
    # eurometro = request.form['eurometro']
    
    # Aquí puedes utilizar los valores recibidos como desees
    # Por ejemplo, imprimirlos en la consola
    print(f"Nombre: {name}")
    print(f"Número de habitaciones: {rooms}")
    print(f"Área en metros cuadrados: {area_m2}")
    print(f"Precio: {prices}")
    print(f"Ascensor: {lift}")
    # print(f"Eurometro: {eurometro}")
    
    return 'Formulario enviado correctamente'

=======
    
    resultado = ia(name, rooms, area_m2, prices, lift)
    # eurometro = request.form['eurometro']
    
    # Llamar a tu función de predicción
#    resultado = tu_funcion_de_prediccion(name, rooms, area_m2, prices, lift, eurometro)
    
    # Renderizar la plantilla 'resultado.html' con el resultado
    return render_template('resultado.html', resultado=resultado)
>>>>>>> 9c57d534ee4a2fe47353fe1379d505e514141a64

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
<<<<<<< HEAD
=======

# from flask import Flask, render_template, jsonify, request
# import os
# import pandas as pd

# app = Flask(__name__)
    
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/procesar', methods=['POST'])
# def procesar():
#     name = request.form['name']
#     rooms = request.form['rooms']
#     area_m2 = request.form['area_m2']
#     prices = request.form['prices']
#     lift = request.form['lift']
#     # eurometro = request.form['eurometro']
    
#     # Aquí puedes utilizar los valores recibidos como desees
#     # Por ejemplo, imprimirlos en la consola
#     print(f"Nombre: {name}")
#     print(f"Número de habitaciones: {rooms}")
#     print(f"Área en metros cuadrados: {area_m2}")
#     print(f"Precio: {prices}")
#     print(f"Ascensor: {lift}")
#     # print(f"Eurometro: {eurometro}")
    
#     return 'Formulario enviado correctamente'


# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 5000))
#     app.run(debug=True, host='0.0.0.0', port=port)
>>>>>>> 9c57d534ee4a2fe47353fe1379d505e514141a64
# from sklearn.linear_model import LogisticRegression
# import joblib
# Cargar el modelo desde el archivo
# loaded_model = joblib.load('modelo_logistic_regression.joblib')

# # Datos de entrada para hacer la predicción
# rooms = 2
# area_m2 = 200
# prices = 100000
# lift = 1
# eurometro = prices / area_m2

# # Preparar los datos en un formato adecuado para hacer la predicción
# new_data = [[rooms, area_m2, prices, lift, eurometro]]

# # Hacer la predicción utilizando el modelo cargado
# prediction = loaded_model.predict(new_data)

# # Obtener el resultado de la predicción
# if prediction[0] == 0:
#     resultado = "Compra recomendable"
# else:
#     resultado = "Compra no recomendable"

# # Imprimir el resultado
# print(resultado)

# Cargar los datos de entrenamiento


# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# import joblib
# import numpy as np

# iris = load_iris()
# X_train = iris.data
# y_train = iris.target

# # Instanciar y ajustar el modelo
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)

# # Guardar el modelo entrenado
# #joblib.dump(lr, 'modelo_entrenado.joblib')

# # Cargar el modelo entrenado
# loaded_model = joblib.load('modelo_logistic_regression.joblib')



# # Datos para hacer la predicción
# new_data = {
#     #"name": ["1"],
#     "rooms": ["2"],
#     "area_m2": ["200"],
#     "prices": ["300000"],
#     #"lift": [1],
#     "eurometro" : ["1500"]
# }

# # Convertir los valores a tipos numéricos
# new_data = {key: np.array(value).astype(float) for key, value in new_data.items()}

# # # Convertir los datos a una matriz 2D
# X_new = np.array(list(new_data.values())).reshape(1, -1)

# # Realizar la predicción
# prediction = loaded_model.predict(X_new)

# # Imprimir la predicción
# print(prediction)
    
