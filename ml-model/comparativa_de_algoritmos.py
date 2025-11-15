# Este script fue implementado y ejecutado en Google Colab


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import dump

# 1. Carga y preprocesamiento de datos (común para todos los modelos)
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['Sentence'] = data['Sentence'].apply(lambda x: str(x).lower())
    return data

# 2. Feature engineering para SQL Injection
def add_sqli_features(X):
    return pd.DataFrame({
        'length': X.apply(len),
        'num_special_chars': X.apply(lambda x: sum(1 for c in x if not c.isalnum())),
        'num_spaces': X.apply(lambda x: x.count(' ')),
        'num_sql_keywords': X.apply(lambda x: sum(x.count(keyword) for keyword in
                                   ['select', 'union', 'insert', 'delete', 'drop', '1=1', '--']))
    })

# 3. Configuración de modelos
def get_xgb_model():
    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

def get_lgbm_model():
    return LGBMClassifier(
        objective='binary',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

def build_cnn_model(input_dim, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_dim, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Pipeline para modelos tradicionales (XGBoost y LightGBM)
def train_traditional_model(model_type, X_train_final, y_train, X_test_final, y_test):
    # Conversión específica para LightGBM
    if model_type == 'lightgbm':
        X_train_final = X_train_final.tocsr() if hasattr(X_train_final, 'tocsr') else X_train_final
        X_test_final = X_test_final.tocsr() if hasattr(X_test_final, 'tocsr') else X_test_final

        # Asegurar tipos float para LightGBM
        X_train_final = X_train_final.astype(np.float32)
        X_test_final = X_test_final.astype(np.float32)

    if model_type == 'xgboost':
        model = get_xgb_model()
        model_name = "XGBoost"
    elif model_type == 'lightgbm':
        model = get_lgbm_model()
        model_name = "LightGBM"

    print(f"\nEntrenando {model_name}...")
    start_time = time.time()
    model.fit(X_train_final, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n Resultados {model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Tiempo entrenamiento: {train_time:.2f}s")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc, f1, recall, train_time

# 5. Pipeline para modelos de deep learning (CNN y LSTM)
def train_dl_model(model_type, X_train, y_train, X_test, y_test):
    # Tokenización para modelos DL
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_len = 100  # Longitud máxima de secuencia
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    if model_type == 'cnn':
        model = build_cnn_model(input_dim=len(tokenizer.word_index)+1, max_len=max_len)
        model_name = "CNN"
    elif model_type == 'lstm':
        model = build_lstm_model(input_dim=len(tokenizer.word_index)+1, max_len=max_len)
        model_name = "LSTM"

    print(f"\nEntrenando {model_name}...")
    start_time = time.time()
    history = model.fit(
        X_train_pad, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test_pad, y_test),
        verbose=1
    )
    train_time = time.time() - start_time

    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n🔍 Resultados {model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Tiempo entrenamiento: {train_time:.2f}s")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc, f1, recall, train_time, tokenizer

# 6. Pipeline completo comparativo
def compare_models(data_path, test_size=0.2):
    # Carga y división de datos
    data = load_data(data_path)
    X = data['Sentence']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Preprocesamiento para modelos tradicionales
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    X_train_features = add_sqli_features(X_train)
    X_test_features = add_sqli_features(X_test)

    X_train_final = hstack([X_train_vec, X_train_features])
    X_test_final = hstack([X_test_vec, X_test_features])

    # Balanceo
    if sum(y_train) / len(y_train) < 0.3:
        smote = SMOTE(random_state=42)
        X_train_final, y_train = smote.fit_resample(X_train_final, y_train)

    # Entrenamiento y evaluación de todos los modelos
    results = []

    # XGBoost
    xgb_model, xgb_acc, xgb_f1, xgb_recall, xgb_time = train_traditional_model(
        'xgboost', X_train_final, y_train, X_test_final, y_test)
    results.append(('XGBoost', xgb_acc, xgb_f1, xgb_recall, xgb_time))

    # LightGBM
    lgbm_model, lgbm_acc, lgbm_f1, lgbm_recall, lgbm_time = train_traditional_model(
        'lightgbm', X_train_final, y_train, X_test_final, y_test)
    results.append(('LightGBM', lgbm_acc, lgbm_f1, lgbm_recall, lgbm_time))

    # CNN
    cnn_model, cnn_acc, cnn_f1, cnn_recall, cnn_time, _ = train_dl_model(
        'cnn', X_train, y_train, X_test, y_test)
    results.append(('CNN', cnn_acc, cnn_f1, cnn_recall, cnn_time))

    # LSTM
    lstm_model, lstm_acc, lstm_f1, lstm_recall, lstm_time, _ = train_dl_model(
        'lstm', X_train, y_train, X_test, y_test)
    results.append(('LSTM', lstm_acc, lstm_f1, lstm_recall, lstm_time))

    # Resultados comparativos
    print("\n Comparativa Final de Modelos:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Modelo', 'Accuracy', 'F1-Score', 'Recall', 'Tiempo (s)'))
    for name, acc, f1, recall, t in results:
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.2f}".format(name, acc, f1, recall, t))

    return {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'cnn_model': cnn_model,
        'lstm_model': lstm_model,
        'vectorizer': vectorizer,
        'results': results
    }

# 7. Ejemplo de uso
if __name__ == "__main__":
    # Comparar todos los modelos
    # Agregue la ubicacion del dataset
    models = compare_models("ubicacion del dataset")

    # Exportar modelos
    dump(models['xgb_model'], 'xgboost_sqli_detector.joblib')
    dump(models['lgbm_model'], 'lightgbm_sqli_detector.joblib')
    dump(models['vectorizer'], 'count_vectorizer.joblib')
    models['cnn_model'].save('cnn_sqli_detector.h5')
    models['lstm_model'].save('lstm_sqli_detector.h5')

    print("\n Modelos exportados:")
    print("- xgboost_sqli_detector.joblib")
    print("- lightgbm_sqli_detector.joblib")
    print("- count_vectorizer.joblib")
    print("- cnn_sqli_detector.h5")
    print("- lstm_sqli_detector.h5")

