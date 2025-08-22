from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- CONFIGURACIÓN DE TU DATASET ---
# Ruta base para el dataset, siguiendo tu convención.
BASE_PATH = 'datasets/datasets/trec07p/'
INDEX_FILE = os.path.join(BASE_PATH, 'full/index')

def load_your_dataset():
    """Carga los datos de texto y las etiquetas desde el archivo de índice y los archivos de correo electrónico."""
    texts = []
    labels = []
    
    try:
        with open(INDEX_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    label, file_path_relative = parts
                    # Construir la ruta completa como en tu ejemplo
                    full_path = os.path.join(BASE_PATH, file_path_relative.replace('../', ''))
                    
                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='latin-1', errors='ignore') as email_file:
                            texts.append(email_file.read())
                            labels.append(1 if label == 'spam' else 0)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de índice en {INDEX_FILE}. Revisa la ruta.")
        return None, None
    except Exception as e:
        print(f"Error al leer el dataset: {e}")
        return None, None

    return texts, np.array(labels)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    # Cargar y preprocesar el dataset
    texts, labels = load_your_dataset()
    if texts is None or not texts:
        return jsonify({'error': 'No se pudo cargar el dataset o está vacío. Revisa las rutas de los archivos.'}), 500

    try:
        n_samples = int(request.json['n_samples'])
        if n_samples < 50:
            return jsonify({'error': 'El número de muestras debe ser al menos 50 para la validación cruzada.'}), 400
    except (ValueError, KeyError):
        return jsonify({'error': 'Número de muestras inválido'}), 400
    
    # Selecciona un subconjunto aleatorio de datos
    indices = np.random.choice(len(texts), n_samples, replace=False)
    selected_texts = [texts[i] for i in indices]
    selected_labels = labels[indices]

    # Vectorización del texto (extracción de características)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(selected_texts)
    y = selected_labels

    # Entrenar el modelo con validación cruzada
    model = SVC(kernel='linear', probability=True)
    
    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # Para las visualizaciones, usamos una división simple
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcular métricas para la visualización
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Generar gráficos
    roc_img = generate_roc_curve(y_test, y_prob)
    pr_img = generate_pr_curve(y_test, y_prob)

    return jsonify({
        'accuracy': np.mean(accuracy_scores),
        'f1_score': np.mean(f1_scores),
        'confusion_matrix': conf_matrix,
        'roc_curve': roc_img,
        'pr_curve': pr_img
    })

def generate_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_pr_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva de Precisión-Recall')
    plt.legend(loc="lower right")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)