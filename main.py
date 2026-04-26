import os
import pandas as pd
import requests
import base64
import glob
import matplotlib
matplotlib.use('Agg') # Obligatorio para servidores sin pantalla
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- CARGA Y LIMPIEZA ---
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
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json()['data']['url']
    except: return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # Forzamos la creación absoluta de la carpeta
        charts_path = os.path.join(os.getcwd(), "exports")
        if not os.path.exists(charts_path):
            os.makedirs(charts_path, exist_ok=True)
        
        # Limpieza total de basura previa
        for f in glob.glob(os.path.join(charts_path, "*.png")):
            os.remove(f)

        # Configuración de Agente con modo 'verbose' para ver fallos en logs
        agent = Agent(
            df, 
            config={
                "llm": llm_instance, 
                "save_charts": True, 
                "save_charts_path": charts_path,
                "verbose": True
            }
        )
        
        # Prompt técnico para obligar a usar matplotlib y guardar
        instruccion = (
            f"Usando matplotlib, genera un gráfico de barras de {request.prompt}. "
            f"Es obligatorio que el gráfico se guarde como un archivo PNG en la carpeta {charts_path}."
        )
        
        print(f"LOG: Intentando generar gráfico en {charts_path}...")
        agent.chat(instruccion)
        
        # Verificación manual del archivo
        generated_files = glob.glob(os.path.join(charts_path, "*.png"))
        
        if generated_files:
            # Ordenamos por fecha para enviar el más reciente
            latest_file = max(generated_files, key=os.path.getctime)
            print(f"LOG: ¡Archivo encontrado!: {latest_file}")
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico generado con éxito"}
        
        print("LOG ERROR: PandasAI terminó pero el archivo no existe en el disco.")
        return {"chart_url": None, "detail": "La IA procesó pero el archivo no se escribió en disco."}

    except Exception as e:
        print(f"LOG CRASH: {str(e)}")
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
