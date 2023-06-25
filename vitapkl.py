import pickle

# Abrir el archivo en modo lectura binaria
with open("model/mlp_model.pkl", "rb") as file:
    # Cargar el objeto desde el archivo
    objeto = pickle.load(file)

# Ahora puedes usar el objeto cargado
# Ejemplo: imprimir el contenido del objeto
print(objeto)

