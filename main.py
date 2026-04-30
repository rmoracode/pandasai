import os
import requests
import base64
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()
app = FastAPI()

DB_URL = (
    f"postgresql+psycopg2://postgres:{os.getenv('PG_PASSWORD')}"
    f"@72.61.2.146:5432/ventas_aje"
)
engine = create_engine(DB_URL)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_sql(query: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

def get_schema_info() -> str:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'ventas'
            ORDER BY ordinal_position
        """))
        cols = result.fetchall()
    return "\n".join([f"- {col[0]} ({col[1]})" for col in cols])

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

def generate_sql(prompt: str, schema: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Eres experto en SQL y PostgreSQL.\n"
                f"Tabla: 'ventas' con columnas:\n{schema}\n\n"
                f"Usuario pregunta: \"{prompt}\"\n\n"
                f"Genera SOLO el SQL necesario, sin explicaciones, sin markdown, sin comillas."
            )
        }],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    return sql.replace("```sql", "").replace("```", "").strip()

@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        schema = get_schema_info()
        sql = generate_sql(request.prompt, schema)
        df_result = execute_sql(sql)

        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"El usuario preguntó: \"{request.prompt}\"\n"
                    f"SQL ejecutado: {sql}\n"
                    f"Resultado:\n{df_result.to_string(index=False)}\n\n"
                    f"Responde de forma clara y concisa en español."
                )
            }],
            temperature=0.3
        )
        return {"response": answer.choices[0].message.content.strip()}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}

@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        schema = get_schema_info()
        sql = generate_sql(request.prompt, schema)
        df_result = execute_sql(sql)

        chart_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"El usuario quiere un gráfico sobre: \"{request.prompt}\"\n"
                    f"Datos disponibles:\n{df_result.to_string(index=False)}\n"
                    f"Columnas: {list(df_result.columns)}\n\n"
                    f"Genera código Python con matplotlib.\n"
                    f"REGLAS:\n"
                    f"1. Usa plt.pie() para pastel\n"
                    f"2. Usa plt.plot() para líneas\n"
                    f"3. Usa plt.fill_between() para área\n"
                    f"4. Usa plt.bar() para barras\n"
                    f"5. Guarda en: {charts_dir}/chart.png con plt.savefig()\n"
                    f"6. Incluye plt.tight_layout() antes de guardar\n"
                    f"7. NO uses plt.show()\n"
                    f"8. Responde SOLO con código Python, sin markdown."
                )
            }],
            temperature=0
        )
        chart_code = chart_response.choices[0].message.content.strip()
        chart_code = chart_code.replace("```python", "").replace("```", "").strip()

        exec(chart_code, {"plt": plt, "df": df_result, "pd": pd})

        chart_path = os.path.join(charts_dir, "chart.png")
        if os.path.exists(chart_path):
            url = upload_to_imgbb(chart_path)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}
        return {"chart_url": None, "detail": "La IA no pudo generar la imagen."}
    except Exception as e:
        return {"chart_url": None, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
