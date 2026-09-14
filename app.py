import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
from huggingface_hub import HfFileSystem

# ⚙️ CONFIGURATION
HF_TOKEN = os.getenv("HF_TOKEN", "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")  # 👈 Token Configured
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required 8 Columns
COLUMNS = "mobile, name, fname, address, alt, circle, id, email"

# Global Variables
PARQUET_FILE_URLS = []
con = None

def init_duckdb():
    global con
    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET allow_asterisks_in_http_paths = true;")  # Asterisk error fix
    con.execute("SET memory_limit='200MB';")                  # Render 512MB Safe Lock
    con.execute("SET threads=1;")
    con.execute("SET enable_object_cache=false;")
    con.execute("SET preserve_insertion_order=false;")
    
    if HF_TOKEN:
        con.execute(f"SET http_headers = '{{\"Authorization\": \"Bearer {HF_TOKEN}\"}}';")

def load_split_urls():
    """Hugging Face Bucket se saare split files ke direct links generate karta hai"""
    global PARQUET_FILE_URLS
    try:
        fs = HfFileSystem(token=HF_TOKEN or None)
        bucket_dir = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}"
        
        # Discover all parquet parts in the bucket
        files = fs.ls(bucket_dir, detail=False)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        
        # Direct resolve URLs generate karo
        PARQUET_FILE_URLS = [
            f"https://huggingface.co/buckets/{BUCKET_NAME}/resolve/main/{SPLITS_FOLDER}/{os.path.basename(f)}"
            for f in parquet_files
        ]
        print(f"✅ Successfully loaded {len(PARQUET_FILE_URLS)} split parquet URLs!")
    except Exception as e:
        print(f"⚠️ Warning loading bucket files: {e}")
        # Fallback URL format
        PARQUET_FILE_URLS = [
            f"https://huggingface.co/buckets/{BUCKET_NAME}/resolve/main/{SPLITS_FOLDER}/*.parquet"
        ]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server start hone par DuckDB aur URLs load karo
    init_duckdb()
    load_split_urls()
    yield
    # Server shutdown cleanup
    if con:
        con.close()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="DuckDB Parquet Search Engine (512MB RAM Safe)",
    version="2.1",
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
        "loaded_splits_count": len(PARQUET_FILE_URLS),
        "memory_limit": "200MB (Render Safe)"
    }

# 🔍 SEARCH ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="10-digit mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    if not PARQUET_FILE_URLS:
        raise HTTPException(status_code=500, detail="Parquet URLs not loaded yet.")

    try:
        cursor = con.cursor()
        
        # Files list ko formatted string me convert karo
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
