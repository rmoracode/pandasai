import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe 
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Carga de datos
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

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    agent = SmartDataframe(df, config={"llm": llm_instance, "enable_cache": False})
    response = agent.chat(request.prompt)
    return {"response": str(response)}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # 1. Crear carpetas de exportación explícitamente
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # 2. Configuración del Agente
        agent = SmartDataframe(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir,
                "verbose": True
            }
        )

        # Limpiar archivos viejos
        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        # 3. REGLA DE ORO: Forzamos el prompt para que la IA dibuje
        # Limpiamos el prompt por si n8n manda datos extra
        user_query = request.prompt.split('\n')[0] 
        instruccion_forzada = f"Genera un gráfico de barras de: {user_query}. Es obligatorio usar matplotlib y guardar el archivo .png"

        # Ejecutamos el chat
        agent.chat(instruccion_forzada)
        
        # 4. Buscamos el archivo generado (independientemente de lo que responda la IA)
        generated_files = glob.glob(os.path.join(charts_dir, "*.png"))
        
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}

        return {"chart_url": None, "detail": "La IA no generó el archivo de imagen. Intenta ser más específico."}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
