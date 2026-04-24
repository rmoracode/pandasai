import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import SmartDataframe
# CAMBIO AQUÍ: Nueva forma de importar OpenAI en PandasAI 2.0+
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Carga de datos
try:
    # Ajusta el separador si tu CSV usa comas o puntos y coma
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# Limpieza básica
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# Configuración del LLM (Sintaxis 2.0)
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
agent = SmartDataframe(df, config={"llm": llm})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # El método chat sigue igual
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
