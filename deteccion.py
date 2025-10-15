# 1. Importar librerías
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Crear dataset
datos = {
    "texto": [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    "etiqueta": [
        "real", "fake", "real", "fake", "real", 
        "fake", "real", "fake", "real", "fake"
    ]
}

df = pd.DataFrame(datos)

# 3. Vectorización de texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["texto"])
y = df["etiqueta"]

# 4. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Entrenar modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = modelo.predict(X_test)

print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred, labels=["real", "fake"]))

# 7. Clasificación de nuevas noticias
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

X_nuevos = vectorizer.transform(nuevas_noticias)
predicciones = modelo.predict(X_nuevos)

print("\nClasificación de nuevas noticias:")
for noticia, etiqueta in zip(nuevas_noticias, predicciones):
    print(f"'{noticia}' -> {etiqueta}")