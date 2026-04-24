import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import SmartDataframe
# IMPORTANTE: Usamos esta ruta que es la más estable en versiones 2.0+
from pandasai.llm import OpenAI 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Carga de datos
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# Configuración del LLM - CREAMOS LA INSTANCIA PRIMERO
# Esto soluciona el error "Input should be an instance of LLM" que vimos en los logs
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Pasamos la instancia directamente, NO un diccionario
agent = SmartDataframe(df, config={"llm": llm_instance})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
