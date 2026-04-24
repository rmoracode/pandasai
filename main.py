import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# RUTAS MODERNAS PARA PANDASAI 3.0+
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

# 3. Creación directa de la INSTANCIA del modelo (Evita el error de Pydantic en los logs)
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# 4. En versión 3.0+ usamos Agent y le pasamos la instancia
agent = Agent(df, config={"llm": llm_instance})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # Ejecutamos la consulta natural
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
