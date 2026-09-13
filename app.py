import os
import shutil
import duckdb
from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import HfFileSystem

app = FastAPI(title="Paytm Search API (Public) ⚡", version="1.0")

# Public Bucket Parquet Path (Token ki koi zaroorat nahi)
BUCKET_PARQUET_PATH = "buckets/Zerotracelegit/paytm/user.parquet"
LOCAL_PARQUET_PATH = "/tmp/user.parquet"

# DuckDB setup
con = duckdb.connect(database=":memory:")
# Render 512MB RAM safety limit
con.execute("SET memory_limit='350MB';")

@app.on_event("startup")
def load_data():
    """App start hote hi public parquet file download & load hogi"""
    print("⏳ Downloading public parquet file...")
    if not os.path.exists(LOCAL_PARQUET_PATH):
        # Anonymous file access (No Token Needed)
        fs = HfFileSystem()
        with fs.open(BUCKET_PARQUET_PATH, "rb") as remote_f, open(LOCAL_PARQUET_PATH, "wb") as local_f:
            shutil.copyfileobj(remote_f, local_f)
            
    # DuckDB In-Memory View create karein
    con.execute(f"CREATE OR REPLACE VIEW users AS SELECT * FROM read_parquet('{LOCAL_PARQUET_PATH}');")
    print("✅ Parquet Loaded! API is Ready to search.")

@app.get("/")
def home():
    return {
        "status": "Online 🟢",
        "message": "Public Paytm Search API is Live!",
        "docs": "/docs"
    }

@app.get("/search")
def search_records(
    query: str = Query(..., description="Search value (Phone, Name, ID, etc.)"),
    field: str = Query("all", description="Search in 'all' columns or specific column"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return (1-100)")
):
    try:
        cols_info = con.execute("PRAGMA table_info('users');").fetchall()
        all_cols = [c[1] for c in cols_info]

        if field == "all":
            # Sabhi columns me search karega
            filters = [f"CAST({col} AS VARCHAR) ILIKE '%{query}%'" for col in all_cols]
            sql = f"SELECT * FROM users WHERE {' OR '.join(filters)} LIMIT {limit};"
        else:
            if field not in all_cols:
                raise HTTPException(status_code=400, detail=f"Column '{field}' not found. Available: {all_cols}")
            sql = f"SELECT * FROM users WHERE CAST({field} AS VARCHAR) ILIKE '%{query}%' LIMIT {limit};"

        # Super-fast execution
        results = con.execute(sql).df().to_dict(orient="records")

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/columns")
def get_columns():
    """Available table columns dekhne ke liye"""
    cols_info = con.execute("PRAGMA table_info('users');").fetchall()
    return {"columns": [c[1] for c in cols_info]}
