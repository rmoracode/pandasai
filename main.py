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

# Instancia del LLM
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

def get_prepared_df(query):
    # Cargamos datos usando el filtro SQL directamente para optimizar
    df = pd.read_sql(query, engine)
    
    # Normalización de sucursales a MAYÚSCULAS para que coincidan con el prompt
    if 'desc_sucursal' in df.columns:
        df['desc_sucursal'] = df['desc_sucursal'].astype(str).str.upper().str.strip()
    
    # Convertimos la columna fecha a datetime de Pandas (Formato M/D/YYYY)
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        
    return df

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        # Filtro directo para Marzo 2026 en formato M/D/YYYY
        query = "SELECT * FROM ventas WHERE fecha LIKE '3/%/2026'"
        df = get_prepared_df(query)
        
        if df.empty:
            return {"response": "No se encontraron datos para los filtros aplicados."}

        # Usar el DataFrame precargado evita el error de SQLAlchemy
        agent = SmartDataframe(df, config={"llm": llm_instance, "enable_cache": False})
        
        instruccion = f"{request.prompt}. Nota: Las sucursales están en MAYÚSCULAS."
        response = agent.chat(instruccion)
        return {"response": str(response)}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Precarga filtrada para gráficos rápidos y sin errores de conexión
        query = "SELECT * FROM ventas WHERE fecha LIKE '3/%/2026'"
        df = get_prepared_df(query)

        agent = SmartDataframe(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir,
                "verbose": True
            }
        )

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        # Forzamos el tipo de gráfico (pie, line, bar) solicitado
        instruccion_forzada = (
            f"{request.prompt}. "
            "IMPORTANTE: Usa estrictamente el tipo de gráfico solicitado (ej. pastel = plt.pie, líneas = plt.plot). "
            "Las sucursales están en MAYÚSCULAS. Guarda como .png."
        )
        agent.chat(instruccion_forzada)
        
        generated_files = glob.glob(os.path.join(charts_dir, "*.png"))
        
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}

        return {"chart_url": None, "detail": "La IA no pudo generar la imagen."}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Puerto dinámico para Railway
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
