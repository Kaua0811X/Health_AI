from flask import Flask, request, render_template
import joblib
import pandas as pd
from googletrans import Translator
import os

# Inicialização
app = Flask(__name__)
modelo_doenca = joblib.load("modelo/modelo_doença.pkl")
vetor = joblib.load("modelo/vetor.pkl")
translator = Translator()

# Mapeamento de doenças para especialistas
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
        correcao = request.form.get("correcao")  # Correção opcional

        # Traduz sintomas para inglês
        sintomas_en = translator.translate(sintomas, src='pt', dest='en').text

        # Vetoriza e prediz
        vetorizado = vetor.transform([sintomas_en])
        doenca_en = modelo_doenca.predict(vetorizado)[0]
        doenca_pt = translator.translate(doenca_en, src='en', dest='pt').text

        especialista = especialistas.get(doenca_en, "Clínico Geral")

        # Salvar a correção se o usuário preencheu
        if correcao and correcao.strip():
            correcao_en = translator.translate(correcao, src='pt', dest='en').text
            nova_linha = pd.DataFrame([[sintomas_en, correcao_en]], columns=["symptom", "disease"])

            # Cria ou anexa ao arquivo de correções
            caminho_corrigido = "dados/correcoes_usuario.csv"
            if os.path.exists(caminho_corrigido):
                nova_linha.to_csv(caminho_corrigido, mode='a', header=False, index=False)
            else:
                nova_linha.to_csv(caminho_corrigido, mode='w', header=True, index=False)

        # Resultado enviado ao HTML
        resultado = {
            "sintomas": sintomas,
            "doenca_pt": doenca_pt,
            "especialista": especialista
        }

    return render_template("index.html", resultado=resultado, **(resultado or {}))


if __name__ == "__main__":
    app.run(debug=True)
