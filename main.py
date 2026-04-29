import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe 
from pandasai_openai import OpenAI 
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Configuración de Base de Datos
user = "postgres"
password = os.getenv("PG_PASSWORD")
host = "72.61.2.146"
port = "5432"
db = "ventas_aje"

db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
engine = create_engine(db_url)

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
    except: return None

class QueryRequest(BaseModel):
    prompt: str

def get_prepared_df():
    # Cargamos la tabla completa
    df = pd.read_sql("SELECT * FROM ventas", engine)
    
    # 1. CORRECCIÓN DE FECHA: Forzamos día/mes/año
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    
    # 2. ESTANDARIZACIÓN A MAYÚSCULAS:
    # Convertimos la columna de la DB a mayúsculas y quitamos espacios
    if 'desc_sucursal' in df.columns:
        df['desc_sucursal'] = df['desc_sucursal'].astype(str).str.upper().str.strip()
    
    return df

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        df = get_prepared_df()
        agent = SmartDataframe(df, config={"llm": llm_instance, "enable_cache": False})
        
        # Forzamos a la IA a que busque siempre en MAYÚSCULAS
        prompt_ajustado = (
            f"Responde a: {request.prompt}. "
            "IMPORTANTE: Las sucursales en los datos están en MAYÚSCULAS. "
            "Asegúrate de convertir cualquier nombre de sucursal en tu código a MAYÚSCULAS antes de filtrar."
        )
        
        response = agent.chat(prompt_ajustado)
        return {"response": str(response)}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        df = get_prepared_df()

        agent = SmartDataframe(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir
            }
        )

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        # También forzamos mayúsculas para los gráficos
        prompt_grafico = f"{request.prompt}. Las sucursales están en MAYÚSCULAS. Usa matplotlib."
        agent.chat(prompt_grafico)
        
        generated_files = glob.glob(os.path.join(charts_dir, "*.png"))
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Éxito"}

        return {"chart_url": None, "detail": "No se generó imagen"}
    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
