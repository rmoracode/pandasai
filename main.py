import os
import pandas as pd
import requests
import base64
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# 3. Configuración del LLM
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Función auxiliar para subir imágenes a ImgBB con logs detallados
def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        print("LOG: Error - Falta la variable IMGBB_API_KEY")
        return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key": api_key,
                "image": base64.b64encode(file.read()),
            }
            res = requests.post(url, payload)
            res_json = res.json()
            if res_json.get("success"):
                print(f"LOG: Imagen subida con éxito: {res_json['data']['url']}")
                return res_json["data"]["url"]
            else:
                print(f"LOG: Error de ImgBB: {res_json}")
                return None
    except Exception as e:
        print(f"LOG: Error subiendo imagen: {e}")
        return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_aje(request: QueryRequest):
    try:
        # 4. Preparación de carpeta y limpieza de basura anterior
        charts_path = "exports"
        if not os.path.exists(charts_path):
            os.makedirs(charts_path)
        
        for f in glob.glob(f"{charts_path}/*.png"):
            os.remove(f)

        # AGENTE REFORZADO: Añadimos instrucciones explícitas para NO usar solo SQL
        agent = Agent(
            df, 
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_path,
                "cache": False,
                "verbose": True,
                "custom_instructions": "If the user asks for a chart, graph, bar chart or plot, you MUST use matplotlib to generate it and save it. Do not just return the data or SQL query."
            }
        )

        # REFUERZO DE PROMPT: Si detectamos intención de graficar, inyectamos la orden técnica
        user_prompt = request.prompt
        if any(word in user_prompt.lower() for word in ["gráfico", "grafica", "barras", "dibujar", "visualiza"]):
            user_prompt += ". Importante: Genera un gráfico de barras usando matplotlib y guárdalo."

        print(f"LOG: Procesando prompt: {user_prompt}")
        answer = agent.chat(user_prompt)
        
        # 5. Lógica de detección de gráficos dinámica
        chart_url = None
        generated_files = glob.glob(f"{charts_path}/*.png")
        
        if generated_files:
            print(f"LOG: Gráfico generado encontrado en: {generated_files[0]}")
            chart_url = upload_to_imgbb(generated_files[0])
        else:
            print("LOG: PandasAI no generó ningún archivo de imagen para este prompt.")

        return {
            "response": str(answer),
            "chart_url": chart_url
        }

    except Exception as e:
        print(f"LOG: Error detectado: {e}")
        return {"response": f"Error interno: {str(e)}", "chart_url": None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
