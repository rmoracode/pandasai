import os
import requests
import base64
import glob
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import Agent
from pandasai_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

DB_URL = (
    f"postgresql+psycopg2://postgres:{os.getenv('PG_PASSWORD')}"
    f"@72.61.2.146:5432/ventas_aje"
)
engine = create_engine(DB_URL)
llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

def get_schema_sample() -> pd.DataFrame:
    """
    En lugar de cargar 1M de filas, trae solo 50 filas para que
    el agente conozca el esquema y los tipos de datos, luego genera
    SQL que se ejecuta directo en Postgres.
    """
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM ventas LIMIT 50"), conn)

def run_sql_on_postgres(sql: str) -> pd.DataFrame:
    """Ejecuta el SQL generado por el agente directo en Postgres."""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

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

def build_agent(extra_config: dict = {}) -> Agent:
    sample_df = get_schema_sample()
    config = {
        "llm": llm_instance,
        "enable_cache": False,
        "description": (
            "Eres un analista de datos. El DataFrame que ves es solo una muestra "
            "de 50 filas para conocer el esquema. La tabla real 'ventas' en PostgreSQL "
            "tiene millones de registros. Para responder consultas de agregación, "
            "totales, conteos o cualquier cálculo, genera SQL eficiente que opere "
            "sobre los datos disponibles."
        ),
        **extra_config
    }
    return Agent([sample_df], config=config)

    # Inyectamos contexto: le decimos que el sample es solo el esquema
    # y que debe ejecutar el SQL real contra la BD completa
    agent.context.memory.add(
        "INSTRUCCIÓN DEL SISTEMA: El DataFrame que ves es solo una muestra "
        "de 50 filas para conocer el esquema. La tabla real 'ventas' en PostgreSQL "
        "tiene millones de registros. Para responder consultas de agregación, "
        "totales, conteos o cualquier cálculo, SIEMPRE genera y ejecuta SQL "
        "directo contra la base de datos usando sqlalchemy con el engine disponible. "
        "Nunca asumas que el sample es la data completa.",
        is_user=False
    )
    return agent, sample_df

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        agent = build_agent()
        response = agent.chat(request.prompt)
        return {"response": str(response)}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        agent = build_agent({
            "save_charts": True,
            "save_charts_path": charts_dir,
            "verbose": True,
        })

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
