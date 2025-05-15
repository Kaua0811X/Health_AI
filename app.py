from flask import Flask, request, render_template
import joblib
from googletrans import Translator

# Inicialização
app = Flask(__name__)
modelo_doenca = joblib.load("modelo/modelo_doença.pkl")
vetor = joblib.load("modelo/vetor.pkl")
translator = Translator()

# Dicionário de exemplo com mapeamento de doença para especialista
especialistas = {
    "Flu": "Clínico Geral",
    "Diabetes": "Endocrinologista",
    "Hypertension": "Cardiologista",
    "Asthma": "Pneumologista",
    "Covid-19": "Infectologista",
    "Migraine": "Neurologista",
    "Depression": "Psiquiatra",
    # Adicione mais se quiser
}

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    sintomas = ""

    if request.method == "POST":
        sintomas = request.form["sintomas"]

        # Traduz os sintomas para inglês
        sintomas_en = translator.translate(sintomas, src='pt', dest='en').text

        # Vetoriza e faz predição
        vetorizado = vetor.transform([sintomas_en])
        doenca_en = modelo_doenca.predict(vetorizado)[0]

        # Traduz a doença para português
        doenca_pt = translator.translate(doenca_en, src='en', dest='pt').text

        # Obtém especialista com fallback
        especialista = especialistas.get(doenca_en, "Clínico Geral")

        resultado = {
            "sintomas": sintomas,
            "doenca": doenca_pt,
            "especialista": especialista
        }

    return render_template("index.html", resultado=resultado, **(resultado or {}))

if __name__ == "__main__":
    app.run(debug=True)
