import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe 
from pandasai_openai import OpenAI 
from pandasai.connectors import PostgreSQLConnector # <-- NUEVO
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Configuración del Conector a Postgres (Hostinger)
# Sustituimos pd.read_csv por este conector inteligente
db_connector = PostgreSQLConnector(config={
    "host": "72.61.2.146",
    "port": 5432,
    "database": "ventas_aje",
    "username": "postgres",
    "password": os.getenv("PG_PASSWORD"),
    "table": "ventas" # Asegúrate de que este sea el nombre de tu tabla
})

# El LLM se mantiene igual
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
    # Ahora pasamos db_connector en lugar de df
    agent = SmartDataframe(db_connector, config={"llm": llm_instance, "enable_cache": False})
    response = agent.chat(request.prompt)
    return {"response": str(response)}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Configuramos el agente con el conector de base de datos
        agent = SmartDataframe(
            db_connector, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir,
                "verbose": True
            }
        )

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        user_query = request.prompt.split('\n')[0] 
        instruccion_forzada = f"Genera un gráfico de barras de: {user_query}. Es obligatorio usar matplotlib y guardar el archivo .png"

        agent.chat(instruccion_forzada)
        
        generated_files = glob.glob(os.path.join(charts_dir, "*.png"))
        
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}

        return {"chart_url": None, "detail": "La IA no generó el archivo de imagen."}

    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
