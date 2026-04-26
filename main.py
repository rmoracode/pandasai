import os
import pandas as pd
import requests
import base64
import matplotlib
matplotlib.use('Agg') # Fundamental para que Railway no intente abrir ventanas gráficas
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
try:
    df = pd.read_csv('fuente.csv', sep=';')
except Exception:
    df = pd.read_csv('fuente.csv', sep=',', encoding='latin1')

# Limpieza estándar de columnas clave
for col in ['VN', 'Vol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# --- 2. FUNCIÓN DE SUBIDA A IMGBB ---
def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        print("LOG: Falta API KEY de ImgBB")
        return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json()['data']['url']
    except Exception as e:
        print(f"LOG: Error subiendo imagen: {e}")
        return None

class QueryRequest(BaseModel):
    prompt: str

# --- RUTA 1: CONSULTAS DE TEXTO ---
@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        answer = agent.chat(request.prompt)
        return {"response": str(answer)}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# --- RUTA 2: GRÁFICO AUTOMÁTICO (TU IDEA MEJORADA) ---
@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        # 1. Extracción de datos pura. Forzamos a la IA a darnos una tabla.
        agent = Agent(df, config={"llm": llm_instance, "save_charts": False})
        
        # El prompt es ultra-específico para que no intente graficar ella misma
        data_query = (
            f"Extrae los datos para: {request.prompt}. "
            "Devuelve únicamente una tabla con los resultados finales."
        )
        data = agent.chat(data_query)

        # 2. Verificamos si tenemos un DataFrame para dibujar
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Limpieza total de figuras previas en memoria
            plt.close('all')
            plt.figure(figsize=(12, 7))
            
            # Identificamos columnas (X=Categoría, Y=Valor)
            col_x = data.columns[0]
            col_y = data.columns[1]
            
            # Limpieza de seguridad: Forzamos a que Y sea numérico por si la IA trajo basura
            data[col_y] = pd.to_numeric(data[col_y], errors='coerce').fillna(0)
            
            # Ordenamos para que el gráfico se vea bien (Top 10)
            data_sorted = data.sort_values(by=col_y, ascending=False).head(10)
            
            # 3. DIBUJO DIRECTO CON MATPLOTLIB
            plt.bar(data_sorted[col_x].astype(str), data_sorted[col_y], color='#00aaff', edgecolor='black')
            plt.title(f"Visualización: {request.prompt}", fontsize=14, pad=20)
            plt.ylabel(str(col_y))
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()

            # Guardamos el archivo físico
            file_path = "auto_chart.png"
            plt.savefig(file_path)
            plt.close() # Liberamos la memoria del servidor

            # 4. SUBIDA Y RESPUESTA
            url = upload_to_imgbb(file_path)
            if url:
                return {"chart_url": url, "detail": "Gráfico renderizado correctamente por el motor directo."}
            else:
                return {"chart_url": None, "detail": "Gráfico creado pero falló la subida a ImgBB."}
        
        # Si no devolvió DataFrame
        return {"chart_url": None, "detail": "La IA no pudo extraer una tabla de datos válida para graficar."}

    except Exception as e:
        print(f"CRASH LOG: {str(e)}")
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Railway usa la variable PORT
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
