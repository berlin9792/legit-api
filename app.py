import os
import gc
import json
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
HF_TOKEN = os.getenv("HF_TOKEN", "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")
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required 8 Columns
COLUMNS = "mobile, name, fname, address, alt, circle, id, email"
# ==============================================================================

PARQUET_FILE_URLS = []
con = None

def init_duckdb():
    global con
    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET allow_asterisks_in_http_paths = true;")
    con.execute("SET memory_limit='200MB';")          # Render 512MB RAM Strict Lock
    con.execute("SET threads=1;")                     # Single thread (Zero RAM Spike)
    con.execute("SET enable_object_cache=false;")
    con.execute("SET preserve_insertion_order=false;")
    
    if HF_TOKEN:
        con.execute(f"SET http_headers = '{{\"Authorization\": \"Bearer {HF_TOKEN}\"}}';")

def fetch_bucket_file_urls():
    """Python standard library se directly HF Bucket ke saare split files list karta hai"""
    global PARQUET_FILE_URLS
    api_url = f"https://huggingface.co/api/buckets/{BUCKET_NAME}/tree/{SPLITS_FOLDER}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            items = json.loads(response.read().decode())
            
            urls = []
            for item in items:
                path = item.get("path", item.get("name", ""))
                if path.endswith(".parquet"):
                    urls.append(f"https://huggingface.co/buckets/{BUCKET_NAME}/resolve/main/{path}")
            
            if urls:
                PARQUET_FILE_URLS = sorted(urls)
                print(f"✅ Successfully loaded {len(PARQUET_FILE_URLS)} split parquet URLs!")
                return
    except Exception as e:
        print(f"ℹ️ Direct API listing notice: {e}")

    # Fallback: Agar API na mile toh standard resolve pattern
    if not PARQUET_FILE_URLS:
        PARQUET_FILE_URLS = [
            f"https://huggingface.co/buckets/{BUCKET_NAME}/resolve/main/{SPLITS_FOLDER}/*.parquet"
        ]

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_duckdb()
    fetch_bucket_file_urls()
    yield
    if con:
        con.close()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="Render 512MB RAM Safe DuckDB Parquet Engine",
    version="2.2",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Mobile Lookup API",
        "loaded_splits": len(PARQUET_FILE_URLS),
        "memory_limit": "200MB (Render Safe)"
    }

# 🔍 MAIN SEARCH ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="10-digit mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    if not PARQUET_FILE_URLS:
        raise HTTPException(status_code=500, detail="Parquet URLs not initialized.")

    try:
        cursor = con.cursor()
        
        # Files parameter format
        if len(PARQUET_FILE_URLS) == 1:
            files_param = repr(PARQUET_FILE_URLS[0])
        else:
            files_param = str(PARQUET_FILE_URLS)

        query = f"""
            SELECT {COLUMNS}
            FROM read_parquet({files_param})
            WHERE CAST(mobile AS VARCHAR) = ?
            LIMIT {limit}
        """
        
        cursor.execute(query, [clean_mobile])
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        results = [dict(zip(col_names, row)) for row in rows]
        gc.collect()

        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": len(results),
            "data": results
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "splits_loaded": len(PARQUET_FILE_URLS)}
