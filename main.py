import os
import requests
import base64
import glob
import pandas as pd
from sqlalchemy import create_engine, text
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

def execute_sql(query: str) -> pd.DataFrame:
    """Ejecuta SQL directo en Postgres y retorna DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

def get_schema_info() -> str:
    """Obtiene columnas y tipos de la tabla ventas."""
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

def smart_query(prompt: str) -> str:
    """
    Usa el LLM para generar SQL, lo ejecuta en Postgres,
    y luego usa el LLM de nuevo para responder en lenguaje natural.
    """
    schema = get_schema_info()

    # Paso 1: pedir al LLM que genere el SQL
    sql_prompt = f"""Eres un experto en SQL y PostgreSQL.
Tienes una tabla llamada 'ventas' con las siguientes columnas:
{schema}

El usuario pregunta: "{prompt}"

Genera ÚNICAMENTE el SQL necesario para responder esta pregunta.
- Usa agregaciones (SUM, COUNT, AVG, GROUP BY) cuando sea necesario
- NO uses LIMIT a menos que el usuario lo pida
- Responde SOLO con el SQL, sin explicaciones, sin markdown, sin comillas
"""

    from openai import OpenAI as OpenAIClient
    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    sql_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0
    )
    sql = sql_response.choices[0].message.content.strip()

    # Limpiar el SQL por si acaso viene con markdown
    sql = sql.replace("```sql", "").replace("```", "").strip()

    # Paso 2: ejecutar el SQL en Postgres
    df_result = execute_sql(sql)

    # Paso 3: pedir al LLM que responda en lenguaje natural con los resultados
    answer_prompt = f"""El usuario preguntó: "{prompt}"

Se ejecutó esta consulta SQL:
{sql}

Y el resultado fue:
{df_result.to_string(index=False)}

Responde al usuario de forma clara y concisa en español, presentando los datos de forma ordenada."""

    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.3
    )
    return answer_response.choices[0].message.content.strip()


def smart_chart(prompt: str, charts_dir: str) -> str | None:
    """Genera SQL, ejecuta, luego crea el gráfico con matplotlib."""
    schema = get_schema_info()

    sql_prompt = f"""Eres un experto en SQL y PostgreSQL.
Tienes una tabla llamada 'ventas' con las siguientes columnas:
{schema}

El usuario quiere un gráfico sobre: "{prompt}"

Genera ÚNICAMENTE el SQL para obtener los datos necesarios para ese gráfico.
- Usa agregaciones apropiadas (SUM, COUNT, AVG, GROUP BY)
- Retorna máximo 2-3 columnas: una para el eje X (categoría) y una o más para el eje Y (valores)
- Responde SOLO con el SQL, sin explicaciones, sin markdown
"""

    from openai import OpenAI as OpenAIClient
    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    sql_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0
    )
    sql = sql_response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    df_result = execute_sql(sql)

    # Pedir al LLM el código matplotlib
    chart_prompt = f"""El usuario quiere un gráfico sobre: "{prompt}"

Los datos disponibles son:
{df_result.to_string(index=False)}

Columnas: {list(df_result.columns)}

Genera código Python con matplotlib para crear este gráfico.
REGLAS:
1. Usa plt.pie() para gráficos de pastel
2. Usa plt.plot() para líneas
3. Usa plt.fill_between() para área
4. Usa plt.bar() para barras
5. Guarda el gráfico en: {charts_dir}/chart.png usando plt.savefig()
6. Incluye plt.tight_layout() antes de guardar
7. NO uses plt.show()
8. Responde SOLO con el código Python, sin markdown
"""

    chart_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": chart_prompt}],
        temperature=0
    )
    chart_code = chart_response.choices[0].message.content.strip()
    chart_code = chart_code.replace("```python", "").replace("```", "").strip()

    # Ejecutar el código del gráfico
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exec(chart_code, {"plt": plt, "df": df_result, "pd": pd})

    chart_path = os.path.join(charts_dir, "chart.png")
    return chart_path if os.path.exists(chart_path) else None


@app.post("/ask")
async def ask_texto(request: QueryRequest):
    try:
        response = smart_query(request.prompt)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error en el servidor: {str(e)}"}


@app.post("/chart")
async def ask_grafico(request: QueryRequest):
    try:
        charts_dir = os.path.join(os.getcwd(), "exports", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        for f in glob.glob(os.path.join(charts_dir, "*.png")):
            os.remove(f)

        chart_path = smart_chart(request.prompt, charts_dir)

        if chart_path:
            url = upload_to_imgbb(chart_path)
            return {"chart_url": url, "detail": "Gráfico generado con éxito."}
        return {"chart_url": None, "detail": "La IA no pudo generar la imagen."}
    except Exception as e:
        return {"chart_url": None, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
