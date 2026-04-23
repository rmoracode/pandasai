import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Carga de datos
df = pd.read_csv('fuente.csv', sep=';')
# Limpieza de datos AJE
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
agent = SmartDataframe(df, config={"llm": llm, "enable_cache": False})

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        response = agent.chat(request.prompt)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
