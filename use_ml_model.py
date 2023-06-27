import pandas as pd
import joblib

def compra(name,rooms,area_m2,prices,lift):

    # Load the saved model
    model = joblib.load('model.pkl')

    # Define the new data
    new_data = {
        "name": [name],
        "rooms": [rooms],
        "area_m2": [area_m2],
        "prices": [prices],
        "lift": [lift],
        "eurometro": [round(int(prices) / int(area_m2))]
    }

    # Create a DataFrame with the new data
    new_df = pd.DataFrame(new_data)

    # Load the saved scaler
    scaler = joblib.load('scaler.pkl')

    # Remove the 'name' column from new_df
    new_df = new_df.drop('name', axis=1)

    # Scale the data
    new_df_scaled = scaler.transform(new_df)

    # Recreate the DataFrame with the scaled columns
    new_df = pd.DataFrame(new_df_scaled, columns=new_df.columns)

    # Make predictions
    y_new = model.predict(new_df)

    # Create a dictionary for mapping the predictions to labels
    leyenda = {1: "Buena compra", 0: "Mala compra"}

    # Get the predicted label
    resultado = leyenda[y_new[0]]

    # Print the result
    return f"{name} es una {resultado}."

# resultado = compra('name', '1', '200', '100000', '1')
# print(resultado)