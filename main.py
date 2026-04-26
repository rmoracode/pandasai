import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- CARGA Y LIMPIEZA DE SIEMPRE ---
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Función ImgBB
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

# --- RUTA 1: SOLO TEXTO (La que ya te funcionaba) ---
@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# --- RUTA 2: SOLO GRÁFICOS (Ruta nueva aislada) ---
@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_path = "exports"
        if not os.path.exists(charts_path): os.makedirs(charts_path)
        for f in glob.glob(f"{charts_path}/*.png"): os.remove(f)

        agent = Agent(df, config={"llm": llm_instance, "save_charts": True, "save_charts_path": charts_path})
        
        # Forzamos prompt para gráfico
        prompt_grafico = f"Genera un gráfico de barras de {request.prompt}. Guarda el archivo PNG."
        agent.chat(prompt_grafico)
        
        generated_files = glob.glob(f"{charts_path}/*.png")
        if generated_files:
            url = upload_to_imgbb(generated_files[0])
            return {"chart_url": url}
        return {"chart_url": None, "detail": "No se pudo generar el archivo de imagen"}
    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
