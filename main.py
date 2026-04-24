import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import SmartDataframe
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Carga de datos con manejo de errores
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# Limpieza de columnas
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# CONFIGURACIÓN UNIVERSAL: Pasamos la API KEY directamente en el config
# Esto evita tener que importar "OpenAI" de rutas que cambian
agent = SmartDataframe(df, config={
    "llm": {
        "type": "openai",
        "api_token": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini"
    }
})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # Usamos .chat() que es el estándar
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        print(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
