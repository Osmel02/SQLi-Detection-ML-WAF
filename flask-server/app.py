from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

model = joblib.load('modelo_entrenado.pkl')
vectorizer = joblib.load('vectorizador.pkl')

def extract_y_preprocessor(solicitud):
    if not solicitud or not isinstance(solicitud, dict):
        return []
    parameters = solicitud['parameters']
    valores_filtrados = [v for k, v in parameters.items() if k not in ['ARG5:Submit','ARG5:Token']]
    print(valores_filtrados)
    return valores_filtrados

@app.route('/predecir', methods=['POST'])
def predecir():
    solicitud = request.get_json()
    print(solicitud)
    data = extract_y_preprocessor(solicitud)
    print(data)

    texto = ' '.join(data).lower()
    X_new = vectorizer.transform([texto])
    print(X_new)
    stexto = ' '.join(data).lower()

    prediccion = model.predict(X_new)[0]
    
    return 'Maliciosa' if prediccion == 1 else 'Legitima'
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)