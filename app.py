import os
import shutil
import duckdb
from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import HfFileSystem

app = FastAPI(title="Paytm Search API ⚡", version="3.1")

# Remote & Local Paths
BUCKET_PARQUET_PATH = "buckets/Zerotracelegit/paytm/user.parquet"
LOCAL_PARQUET_PATH = "/tmp/user.parquet"

# DuckDB Connection
con = duckdb.connect(database=":memory:")
con.execute("SET memory_limit='350MB';")

def get_columns():
    """Table ke columns nikalne ke liye safe function"""
    try:
        cols_data = con.execute("DESCRIBE users;").fetchall()
        return [c[0] for c in cols_data]
    except Exception:
        return []

@app.on_event("startup")
def startup_db():
    print("⏳ Downloading & Loading Parquet into memory...")
    try:
        if not os.path.exists(LOCAL_PARQUET_PATH):
            fs = HfFileSystem()
            with fs.open(BUCKET_PARQUET_PATH, "rb") as r_file, open(LOCAL_PARQUET_PATH, "wb") as l_file:
                shutil.copyfileobj(r_file, l_file)
                
        # Parquet ko In-Memory Table me load karein
        con.execute(f"CREATE OR REPLACE TABLE users AS SELECT * FROM read_parquet('{LOCAL_PARQUET_PATH}');")
        cols = get_columns()
        print(f"✅ Loaded successfully! Columns found: {cols}")
    except Exception as e:
        print(f"❌ Startup Error: {e}")

@app.get("/")
def home():
    cols = get_columns()
    return {
        "status": "Online 🟢",
        "total_columns": len(cols),
        "columns": cols,
        "docs": "/docs"
    }

@app.get("/search")
def search_records(
    query: str = Query(..., description="Search value (e.g. Phone, Name, Email)"),
    field: str = Query("all", description="Search specific column or 'all'"),
    limit: int = Query(20, ge=1, le=100)
):
    try:
        # Columns verify karein
        available_cols = get_columns()
        if not available_cols:
            raise HTTPException(status_code=500, detail="Database table is empty or still loading. Please retry in 10 seconds.")

        # SQL Injection safety / clean input
        clean_q = query.replace("'", "''").strip()

        # CASE 1: Specific Field Search
        if field != "all":
            if field not in available_cols:
                raise HTTPException(status_code=400, detail=f"Column '{field}' not found. Available columns: {available_cols}")
            sql = f'SELECT * FROM users WHERE CAST("{field}" AS VARCHAR) ILIKE \'%{clean_q}%\' LIMIT {limit};'

        # CASE 2: All Fields Search
        else:
            where_clauses = [f'CAST("{col}" AS VARCHAR) ILIKE \'%{clean_q}%\'' for col in available_cols]
            if not where_clauses:
                raise HTTPException(status_code=500, detail="No columns found to query.")
            sql = f'SELECT * FROM users WHERE {" OR ".join(where_clauses)} LIMIT {limit};'

        # Fast execution via PyArrow
        results = con.execute(sql).arrow().to_pylist()

        return {
            "success": True,
            "query": query,
            "field": field,
            "count": len(results),
            "data": results
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Execution Error: {str(e)}")

@app.get("/columns")
def list_columns():
    return {"columns": get_columns()}
