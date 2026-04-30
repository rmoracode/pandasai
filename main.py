import os
import requests
import base64
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDatalake
from pandasai.connectors import PostgreSQLConnector
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Configuración del conector — NO carga datos, solo define la conexión
pg_config = {
    "host": "72.61.2.146",
    "port": 5432,
    "database": "ventas_aje",
    "username": "postgres",
    "password": os.getenv("PG_PASSWORD"),
    "table": "ventas",
}

llm_instance = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

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
        connector = PostgreSQLConnector(config=pg_config)
        agent = SmartDatalake(
            [connector],
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

        connector = PostgreSQLConnector(config=pg_config)
        agent = SmartDatalake(
            [connector],
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
