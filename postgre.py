# from fastapi import FastAPI, HTTPException
# from sqlalchemy import create_engine, text, inspect
# from sqlalchemy.exc import SQLAlchemyError
# from langchain_community.utilities import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import AIMessage
# from pydantic import BaseModel
# import os
# import re
# from fastapi.staticfiles import StaticFiles
# from dotenv import load_dotenv
# from itertools import combinations

# # ✅ Load environment variables
# load_dotenv()

# app = FastAPI()

# app.mount("/images", StaticFiles(directory="/home/jahanzaib/Pictures/Screenshots"), name="images")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# DATABASE_NAME = "company"
# password = "admin"
# user = "postgres"
# host = "localhost"
# port = "5432"

# DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{DATABASE_NAME}"

# def get_db_connection():
#     try:
#         engine = create_engine(DATABASE_URL)
#         print(f"✅ Connected to {DATABASE_NAME} successfully!")
#         return SQLDatabase(engine)
#     except SQLAlchemyError as e:
#         print(f"❌ Database connection error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Database connection failed.")

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD9kmbO735ZRG-Vnk-iegTodps0ASbQq7A")
# llm_gemini = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.1,
#     google_api_key=GOOGLE_API_KEY,
# )

# prompt = PromptTemplate.from_template(
#     "Given the table '{table_name}' with columns {table_columns}, "
#     "write an SQL query to: {question}"
# )

# class QueryRequest(BaseModel):
#     query: str
#     table_name: str

# def extract_sql(response: str) -> str:
#     match = re.search(r"```sql(.*?)```", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     match = re.search(r"(SELECT .*?;)", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return response.strip()

# def get_table_columns(engine, table_name: str):
#     inspector = inspect(engine)
#     columns = inspector.get_columns(table_name)
#     return [col["name"] for col in columns]

# @app.get("/tables/")
# async def get_tables():
#     db = get_db_connection()
#     try:
#         with db._engine.connect() as conn:
#             result = conn.execute(
#                 text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
#             )
#             tables = [row[0] for row in result.fetchall()]
#         return {"tables": ["All Tables"] + tables}
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/ask/")
# async def ask_question(request: QueryRequest):
#     db = get_db_connection()

#     try:
#         if request.table_name == "All Tables":
#             query_lower = request.query.lower().strip()
#             # Check if it's just a request to list tables
#             if query_lower in ["show all tables", "list all tables", "what are the tables", "get tables"]:
#                 with db._engine.connect() as conn:
#                     result = conn.execute(
#                         text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
#                     )
#                     tables = [row[0] for row in result.fetchall()]
#                     all_data = {}
#                     for table in tables:
#                         sql_query = f'SELECT * FROM "{table}" LIMIT 10;'
#                         table_result = conn.execute(text(sql_query))
#                         column_names = table_result.keys()
#                         formatted_data = [dict(zip(column_names, row)) for row in table_result]
#                         all_data[table] = formatted_data
#                 return {"answer": all_data}
#             else:
#                 # Intelligent SQL generation across all tables
#                 inspector = inspect(db._engine)
#                 tables = inspector.get_table_names()
#                 for table in tables:
#                     columns = get_table_columns(db._engine, table)
#                     llm_chain = prompt | llm_gemini
#                     generated_sql = llm_chain.invoke({
#                         "question": request.query,
#                         "table_name": table,
#                         "table_columns": ", ".join(columns),
#                     })
#                     if isinstance(generated_sql, AIMessage):
#                         generated_sql = generated_sql.content
#                     clean_sql = extract_sql(generated_sql)
#                     try:
#                         with db._engine.connect() as conn:
#                             result = conn.execute(text(clean_sql))
#                             column_names = result.keys()
#                             formatted_data = [dict(zip(column_names, row)) for row in result.fetchall()]
#                             if formatted_data:
#                                 return {"answer": {table: formatted_data}}
#                     except Exception:
#                         continue
#                 return {"answer": "Could not match the query across available tables."}

#         else:
#             columns = get_table_columns(db._engine, request.table_name)
#             llm_chain = prompt | llm_gemini
#             generated_sql = llm_chain.invoke({
#                 "question": request.query,
#                 "table_name": request.table_name,
#                 "table_columns": ", ".join(columns),
#             })
#             if isinstance(generated_sql, AIMessage):
#                 generated_sql = generated_sql.content
#             clean_sql = extract_sql(generated_sql)
#             with db._engine.connect() as conn:
#                 result = conn.execute(text(clean_sql))
#                 column_names = result.keys()
#                 formatted_data = [dict(zip(column_names, row)) for row in result.fetchall()]
#                 return {"answer": formatted_data}

#     except SQLAlchemyError as db_error:
#         return {"error": f"Invalid SQL or execution error: {str(db_error)}"}
#     except Exception as e:
#         return {"error": f"General failure: {str(e)}"}


from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel
import os
import re
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from itertools import combinations

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

#app.mount("/images", StaticFiles(directory="/home/jahanzaib/Pictures/Screenshots"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_NAME = "company"
password = "admin"
user = "postgres"
host = "localhost"
port = "5432"

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{DATABASE_NAME}"

def get_db_connection():
    try:
        engine = create_engine(DATABASE_URL)
        print(f"✅ Connected to {DATABASE_NAME} successfully!")
        return SQLDatabase(engine)
    except SQLAlchemyError as e:
        print(f"❌ Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD9kmbO735ZRG-Vnk-iegTodps0ASbQq7A")
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
)

prompt = PromptTemplate.from_template(
    "Given the table '{table_name}' with columns {table_columns}, "
    "write an SQL query to: {question}"
)

class QueryRequest(BaseModel):
    query: str
    table_name: str

def extract_sql(response: str) -> str:
    match = re.search(r"```sql(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT .*?;)", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()

def get_table_columns(engine, table_name: str):
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [col["name"] for col in columns]

@app.get("/tables/")
async def get_tables():
    db = get_db_connection()
    try:
        with db._engine.connect() as conn:
            result = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            )
            tables = [row[0] for row in result.fetchall()]
        return {"tables": ["All Tables"] + tables}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    db = get_db_connection()

    try:
        if request.table_name == "All Tables":
            query_lower = request.query.lower().strip()
            if query_lower in ["show all tables", "list all tables", "what are the tables", "get tables"]:
                with db._engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                    )
                    tables = [row[0] for row in result.fetchall()]
                    all_data = {}
                    for table in tables:
                        sql_query = f'SELECT * FROM "{table}" LIMIT 10;'
                        table_result = conn.execute(text(sql_query))
                        column_names = table_result.keys()
                        formatted_data = [dict(zip(column_names, row)) for row in table_result]
                        all_data[table] = formatted_data
                return {"answer": all_data}
            else:
                inspector = inspect(db._engine)
                tables = inspector.get_table_names()
                for table_combo_len in range(1, len(tables) + 1):
                    for table_combo in combinations(tables, table_combo_len):
                        combo_name = ", ".join(table_combo)
                        combo_columns = []
                        for t in table_combo:
                            cols = get_table_columns(db._engine, t)
                            combo_columns.extend([f"{t}.{col}" for col in cols])
                        llm_chain = prompt | llm_gemini
                        generated_sql = llm_chain.invoke({
                            "question": request.query,
                            "table_name": combo_name,
                            "table_columns": ", ".join(combo_columns),
                        })
                        if isinstance(generated_sql, AIMessage):
                            generated_sql = generated_sql.content
                        clean_sql = extract_sql(generated_sql)
                        try:
                            with db._engine.connect() as conn:
                                result = conn.execute(text(clean_sql))
                                column_names = result.keys()
                                formatted_data = [dict(zip(column_names, row)) for row in result.fetchall()]
                                if formatted_data:
                                    return {"answer": {combo_name: formatted_data}}
                        except Exception:
                            continue
                return {"answer": "Could not match the query across available tables."}

        else:
            columns = get_table_columns(db._engine, request.table_name)
            llm_chain = prompt | llm_gemini
            generated_sql = llm_chain.invoke({
                "question": request.query,
                "table_name": request.table_name,
                "table_columns": ", ".join(columns),
            })
            if isinstance(generated_sql, AIMessage):
                generated_sql = generated_sql.content
            clean_sql = extract_sql(generated_sql)
            with db._engine.connect() as conn:
                result = conn.execute(text(clean_sql))
                column_names = result.keys()
                formatted_data = [dict(zip(column_names, row)) for row in result.fetchall()]
                return {"answer": formatted_data}

    except SQLAlchemyError as db_error:
        return {"error": f"Invalid SQL or execution error: {str(db_error)}"}
    except Exception as e:
        return {"error": f"General failure: {str(e)}"}
