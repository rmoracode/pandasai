import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from pandasai.responses.charts_response import ChartsResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Carga y Limpieza (Sin cambios para no dañar nada)
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key: return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json()['data']['url']
    except Exception as e:
        print(f"LOG Error ImgBB: {e}")
        return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        charts_path = "exports"
        if not os.path.exists(charts_path): os.makedirs(charts_path)
        for f in glob.glob(f"{charts_path}/*.png"): os.remove(f)

        # Configuración de Agente Reforzada
        agent = Agent(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_path,
                "verbose": True,
                "response_converter": ChartsResponse # Forzamos que la respuesta se incline a gráficos
            }
        )

        # Inyección técnica: Le decimos que NO responda con texto si pedimos gráfico
        user_prompt = request.prompt
        intent_chart = any(word in user_prompt.lower() for word in ["gráfico", "grafica", "barras", "visualiza", "dibujar"])
        
        if intent_chart:
            user_prompt = f"Using the dataframe, create a professional bar chart of {user_prompt}. Save the chart as a PNG file. Use matplotlib. Do not return text, only generate the chart file."

        print(f"LOG: Ejecutando: {user_prompt}")
        answer = agent.chat(user_prompt)
        
        chart_url = None
        # Buscamos de forma recursiva por si PandasAI lo guarda en subcarpetas
        generated_files = glob.glob(f"{charts_path}/**/*.png", recursive=True) + glob.glob(f"{charts_path}/*.png")
        
        if generated_files:
            print(f"LOG: ¡GRÁFICO DETECTADO! -> {generated_files[0]}")
            chart_url = upload_to_imgbb(generated_files[0])
        else:
            print("LOG: Sigue sin generarse el archivo físico.")

        return {"response": str(answer), "chart_url": chart_url}

    except Exception as e:
        print(f"LOG ERROR: {e}")
        return {"response": f"Error: {str(e)}", "chart_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
