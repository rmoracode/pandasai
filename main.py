import os
import pandas as pd
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. Carga de datos con manejo de codificación
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# 2. Limpieza de columnas clave
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# 3. Configuración del LLM
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Función auxiliar para subir imágenes a ImgBB
def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key": api_key,
                "image": base64.b64encode(file.read()),
            }
            res = requests.post(url, payload)
            return res.json()["data"]["url"]
    except Exception as e:
        print(f"Error subiendo imagen: {e}")
        return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # 4. Creamos el Agente configurado para guardar gráficos
        # Usamos una subcarpeta 'exports' para los gráficos
        # Crear carpeta de exportación si no existe
        if not os.path.exists("exports"):
            os.makedirs("exports")
        agent = Agent(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": "exports",
                "cache": False # Recomendado para datos que cambian
            }
        )

        # Ejecutamos la consulta
        answer = agent.chat(request.prompt)
        
        # 5. Lógica de detección de gráficos
        chart_url = None
        # PandasAI suele guardar el último gráfico como temp_chart.png o similar en la ruta especificada
        chart_filename = "exports/temp_chart.png" 
        
        if os.path.exists(chart_filename):
            chart_url = upload_to_imgbb(chart_filename)
            os.remove(chart_filename) # Borramos el archivo local para no acumular basura

        return {
            "response": str(answer),
            "chart_url": chart_url
        }

    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
