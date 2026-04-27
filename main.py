import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe # <-- CLAVE: Usamos exactamente lo que usas en Streamlit
from pandasai.llm import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Carga de datos igual que en tu Streamlit
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Función para subir la imagen generada a ImgBB
def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key: return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json().get('data', {}).get('url')
    except:
        return None

class QueryRequest(BaseModel):
    prompt: str

# --- RUTA 1: TEXTO ---
@app.post("/ask")
async def ask_texto(request: QueryRequest):
    agent = SmartDataframe(df, config={"llm": llm_instance, "enable_cache": False})
    response = agent.chat(request.prompt)
    return {"response": str(response)}

# --- RUTA 2: GRÁFICOS (Modo Streamlit) ---
@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # Configuración EXACTA a tu app web
        agent = SmartDataframe(
            df, 
            config={
                "llm": llm_instance,
                "verbose": True,
                "enable_cache": False,
                "save_charts": True # Aseguramos que lo guarde en disco
            }
        )

        # Limpiamos basura vieja antes de preguntar
        default_path = os.path.join(os.getcwd(), "exports", "charts", "*.png")
        for f in glob.glob(default_path):
            try: os.remove(f)
            except: pass

        # 1. Ejecutamos la consulta tal como en Streamlit
        response = agent.chat(request.prompt)
        
        # 2. Lógica idéntica a tu Streamlit: verificamos si devolvió una ruta .png
        if isinstance(response, str) and response.endswith('.png'):
            if os.path.exists(response):
                url = upload_to_imgbb(response)
                return {"chart_url": url, "detail": "Gráfico generado y subido desde ruta directa."}

        # 3. Fallback: Por si no devuelve la ruta exacta pero sí guardó el archivo
        generated_files = glob.glob(default_path)
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico encontrado en la carpeta de exportación."}

        return {"chart_url": None, "detail": f"PandasAI no generó gráfico. Respuesta: {str(response)}"}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
