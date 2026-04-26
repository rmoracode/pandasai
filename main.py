import os
import pandas as pd
import requests
import base64
import matplotlib
matplotlib.use('Agg') # Crucial para servidores
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- CARGA Y LIMPIEZA DE DATOS ---
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

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

# RUTA 1: Texto (Sin cambios, ya funciona)
@app.post("/ask")
async def ask_texto(request: QueryRequest):
    agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
    answer = agent.chat(request.prompt)
    return {"response": str(answer)}

# RUTA 2: Gráfico Automático (Tu idea aplicada)
@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # 1. Le pedimos a la IA SOLO los datos en formato tabla
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        data_query = f"Extrae los datos necesarios para: {request.prompt}. Devuelve solo la tabla de resultados."
        data = agent.chat(data_query)

        # 2. Si la IA devuelve un DataFrame (que es lo que hace cuando filtra datos)
        if isinstance(data, pd.DataFrame):
            # Limpiamos figuras previas
            plt.clf() 
            plt.figure(figsize=(10, 6))
            
            # Dibujamos automáticamente: X es la primera columna, Y es la segunda
            col_x = data.columns[0]
            col_y = data.columns[1]
            
            # Ordenamos para que el gráfico se vea profesional
            data = data.sort_values(by=col_y, ascending=False).head(10) # Top 10 para no saturar
            
            plt.bar(data[col_x].astype(str), data[col_y], color='#00aaff')
            plt.title(f"Análisis Automático: {request.prompt}", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Guardado forzado
            file_path = "auto_chart.png"
            plt.savefig(file_path)
            plt.close()

            url = upload_to_imgbb(file_path)
            return {"chart_url": url, "detail": "Gráfico generado por motor directo"}
        
        return {"chart_url": None, "detail": "La IA no devolvió una tabla de datos válida."}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
