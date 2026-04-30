import os
import requests
import base64
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- Conexión a PostgreSQL via SQLAlchemy ---
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://postgres:{os.getenv('PG_PASSWORD')}@72.61.2.146:5432/ventas_aje"
    )

def load_dataframe(query: str = "SELECT * FROM ventas LIMIT 50000"):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df

# --- LLM ---
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# --- ImgBB upload ---
def upload_to_imgbb(image_path):
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        return None
    try:
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {"key": api_key, "image": base64.b64encode(file.read())}
            res = requests.post(url, payload)
            return res.json().get("data", {}).get("url")
    except Exception:
        return None

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        df = load_dataframe()
        agent = Agent(
            [df],
            config={"llm": llm_instance, "enable_cache": False}
        )
        response = agent.chat(request.prompt)
        return {"response": str(response)}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        df = load_dataframe()
        agent = Agent(
            [df],
            config={
                "llm": llm_instance,
                "save_charts": True,
                "save_charts_path": charts_dir,
                "verbose": True,
                "enable_cache": False,
            }
        )

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        instruccion_forzada = (
            f"{request.prompt}. "
            "REGLAS CRÍTICAS: "
            "1. IDENTIFICA el tipo de gráfico solicitado. "
            "2. Si pide 'PASTEL', usa plt.pie(). "
            "3. Si pide 'LÍNEAS', usa plt.plot(). "
            "4. Si pide 'ÁREA', usa plt.fill_between() o df.plot.area(). "
            "5. Si pide 'BARRAS', usa plt.bar(). "
            "6. PROHIBIDO usar plt.bar() si se pidió pastel o líneas. "
            "7. Usa matplotlib y guarda como .png."
        )

        agent.chat(instruccion_forzada)

        generated_files = glob.glob(os.path.join(charts_dir, "*.png"))
        if generated_files:
            latest_file = max(generated_files, key=os.path.getctime)
            url = upload_to_imgbb(latest_file)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}
        return {"chart_url": None, "detail": "La IA no pudo generar la imagen."}
    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
