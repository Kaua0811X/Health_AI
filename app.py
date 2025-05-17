from flask import Flask, request, render_template
import joblib
import json
import os

# Inicialização
app = Flask(__name__)
modelo_doenca = joblib.load("modelo/modelo_doenca.pkl")

# Carrega dicionário de especialistas gerado no treinamento
with open("modelo/especialistas.json", "r", encoding="utf-8") as f:
    especialistas_dict = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    sintomas = ""

    if request.method == "POST":
        sintomas = request.form["sintomas"]
        correcao = request.form.get("correcao")

        # Predição usando sintomas em português
        doenca_pt = modelo_doenca.predict([sintomas.lower().strip()])[0]
        especialista = especialistas_dict.get(doenca_pt, "Clínico Geral")

        # Armazena correção do usuário (se fornecida)
        if correcao and correcao.strip():
            novo_registro = {
                "sintomas": sintomas,
                "correcao": correcao
            }

            caminho_corrigido = "dados/correcoes_usuario.json"
            if os.path.exists(caminho_corrigido):
                with open(caminho_corrigido, "r", encoding="utf-8") as f:
                    dados_existentes = json.load(f)
            else:
                dados_existentes = []

            dados_existentes.append(novo_registro)

            with open(caminho_corrigido, "w", encoding="utf-8") as f:
                json.dump(dados_existentes, f, ensure_ascii=False, indent=2)

        # Prepara resultado para exibição
        resultado = {
            "sintomas": sintomas,
            "doenca_pt": doenca_pt,
            "especialista": especialista
        }

    return render_template("index.html", resultado=resultado, **(resultado or {}))

if __name__ == "__main__":
    app.run(debug=True)