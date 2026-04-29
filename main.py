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

# 1. Configuración del Motor de Base de Datos (SQLAlchemy)
user = "postgres"
password = os.getenv("PG_PASSWORD")
host = "72.61.2.146"
port = "5432"
db = "ventas_aje"

# Creamos la URL de conexión robusta
db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
engine = create_engine(db_url)

# Instancia del LLM (GPT-4o-mini es excelente para esto)
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
    try:
        # PRECARGA CRÍTICA: Traemos los datos de la tabla 'ventas' a un DataFrame de Pandas
        # Esto soluciona el ValueError: Invalid input data
        df = pd.read_sql("SELECT * FROM ventas WHERE fecha LIKE '3/%/2026'", engine)
        
        # Inicializamos el agente con el DataFrame ya cargado
        agent = SmartDataframe(df, config={"llm": llm_instance, "enable_cache": False})
        
        response = agent.chat(request.prompt)
        return {"response": str(response)}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Precarga de datos para asegurar el flujo de gráficos
        df = pd.read_sql("SELECT * FROM ventas", engine)

        agent = SmartDataframe(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir,
                "verbose": True
            }
        )

        # Limpieza de gráficos anteriores
        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        # Forzamos la instrucción técnica para matplotlib
        instruccion_forzada = (
            f"{request.prompt}. "
            "IMPORTANTE: Debes usar estrictamente el tipo de gráfico solicitado (ej. si pide pastel usa plt.pie, si pide líneas usa plt.plot). "
            "Usa matplotlib y guarda obligatoriamente el archivo como .png."
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
    # Railway usa la variable de entorno PORT
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
