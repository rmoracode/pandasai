import os
import pandas as pd
import requests
import base64
import matplotlib
matplotlib.use('Agg') # Para que funcione en servidor
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Carga de datos
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# Limpieza de VN y Vol
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json()['data']['url']
    except: return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
    answer = agent.chat(request.prompt)
    return {"response": str(answer)}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # PASO A: La IA solo extrae los datos (No piensa en el gráfico)
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        data_query = f"Devuelve solo una tabla con los datos de: {request.prompt}"
        data = agent.chat(data_query)

        if not isinstance(data, pd.DataFrame):
            return {"chart_url": None, "detail": "No se pudieron extraer datos para graficar."}

        # PASO B: Generación automática (Sin que la IA intervenga aquí)
        plt.figure(figsize=(10, 6))
        # Usamos la primera columna para X y la segunda para Y automáticamente
        plt.bar(data.iloc[:, 0].astype(str), data.iloc[:, 1], color='#e63946')
        plt.title(f"Análisis de {request.prompt}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        file_path = "temp_chart.png"
        plt.savefig(file_path)
        plt.close()

        # PASO C: Subir y entregar
        url = upload_to_imgbb(file_path)
        return {"chart_url": url, "response": "Gráfico generado automáticamente."}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
