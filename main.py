import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Intentar cargar el CSV con manejo de errores de encoding
try:
    df = pd.read_csv('fuente.csv', sep=';')
except UnicodeDecodeError:
    df = pd.read_csv('fuente.csv', sep=';', encoding='latin1')

# Limpieza de datos AJE
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# Configuración del LLM
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# En versiones nuevas, SmartDataframe maneja mejor la config así:
agent = SmartDataframe(df, config={"llm": llm})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # Forzamos la respuesta a string para que FastAPI la pueda enviar
        response = agent.chat(request.prompt)
        return {"response": str(response)}
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
