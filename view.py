
from flask import Flask, render_template, request
import os
from use_ml_model import compra

app = Flask(__name__)

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
    
    resultado = compra(name, rooms, area_m2, prices, lift)
    # eurometro = request.form['eurometro']
    
    # Llamar a tu función de predicción
#    resultado = tu_funcion_de_prediccion(name, rooms, area_m2, prices, lift, eurometro)
    
    # Renderizar la plantilla 'resultado.html' con el resultado
    return render_template('resultado.html', resultado=resultado)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)