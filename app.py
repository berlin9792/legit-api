import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
from huggingface_hub import HfFileSystem

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
    con.execute("SET memory_limit='250MB';")          # Strict 250MB RAM limit
    con.execute("SET threads=1;")                     # Single thread for memory safety
    con.execute("SET enable_object_cache=false;")
    con.execute("SET preserve_insertion_order=false;")
    
    if HF_TOKEN:
        con.execute(f"SET http_headers = '{{\"Authorization\": \"Bearer {HF_TOKEN}\"}}';")

def load_bucket_urls():
    """Hugging Face FileSystem se split files ki direct URL list load karta hai"""
    global PARQUET_FILE_URLS
    try:
        fs = HfFileSystem(token=HF_TOKEN)
        # Search path in bucket
        search_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}"
        files = fs.ls(search_path, detail=False)
        
        urls = []
        for f in sorted(files):
            if f.endswith(".parquet"):
                # Clean path to filename
                file_name = f.split("/")[-1]
                direct_url = f"https://huggingface.co/buckets/{BUCKET_NAME}/resolve/main/{SPLITS_FOLDER}/{file_name}"
                urls.append(direct_url)
        
        if urls:
            PARQUET_FILE_URLS = urls
            print(f"✅ Successfully loaded {len(PARQUET_FILE_URLS)} split parquet URLs!")
        else:
            print("⚠️ No .parquet files found in directory.")
            
    except Exception as e:
        print(f"⚠️ Error scanning bucket with HfFileSystem: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_duckdb()
    load_bucket_urls()
    yield
    if con:
        con.close()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="DuckDB Parquet Engine (512MB RAM Safe)",
    version="2.0",
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
        "memory_limit": "250MB (Render Safe)"
    }

# 🔍 MAIN SEARCH ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    if not PARQUET_FILE_URLS:
        # Retry loading if empty
        load_bucket_urls()
        if not PARQUET_FILE_URLS:
            raise HTTPException(status_code=500, detail="Parquet split files not found in bucket.")

    try:
        cursor = con.cursor()
        
        # Pass exact list of URLs directly to DuckDB
        files_param = str(PARQUET_FILE_URLS)
        
        query = f"""
            SELECT {COLUMNS}
            FROM read_parquet({files_param})
            WHERE CAST(mobile AS VARCHAR) = ?
            LIMIT {limit}
        """
        
        cursor.execute(query, [clean_mobile])
        
        # Native zero-memory extraction (No pandas overhead)
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
    return {
        "status": "ok", 
        "splits_loaded": len(PARQUET_FILE_URLS)
    }
