import os
import gc
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem

# ==============================================================================
# ⚙️ CONFIGURATION (Safe Environment Token with Fallback)
# ==============================================================================
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Render Environment se uthayega
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required Columns to return
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
# ==============================================================================

# Initialize FileSystem (Uses token if available, else anonymous)
fs = HfFileSystem(token=HF_TOKEN) if HF_TOKEN else HfFileSystem()
master_index = []
load_error = None

def fetch_master_index():
    """Bucket se direct index.json load karta hai"""
    global master_index, load_error
    index_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/index.json"
    try:
        with fs.open(index_path, "rb") as f:
            content = f.read().decode("utf-8")
            master_index = json.loads(content)
        load_error = None
        print(f"✅ Master Index loaded successfully! ({len(master_index)} splits indexed)")
        return True
    except Exception as e:
        load_error = str(e)
        print(f"❌ Error loading index.json: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    fetch_master_index()
    yield
    gc.collect()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="Indexed Parquet Search Engine (Render 512MB RAM Compliant)",
    version="6.5",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🏠 1. HOME & HEALTH STATUS
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    if not master_index:
        fetch_master_index()
        
    return {
        "status": "online",
        "engine": "Master Indexed Binary Search",
        "total_indexed_files": len(master_index),
        "auth_mode": "Authenticated Token (High Rate-Limit) 🚀" if HF_TOKEN else "Anonymous Public 🌐",
        "index_status": "Ready 🟢" if len(master_index) > 0 else "Loading Error ❌",
        "error_details": load_error,
        "memory_limit": "512MB Safe (< 50MB Active RAM)"
    }

# 🔍 2. MAIN LOOKUP ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    global master_index
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number. Only digits allowed.")

    if not master_index:
        fetch_master_index()
        if not master_index:
            raise HTTPException(
                status_code=500, 
                detail=f"Index failed to load from bucket: {load_error}"
            )

    # ⚡ STEP 1: In-Memory Range Check (0.0001s)
    matching_files = [
        item["file"] for item in master_index 
        if item["min"] <= clean_mobile <= item["max"]
    ]

    if not matching_files:
        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": 0,
            "data": []
        }

    # ⚡ STEP 2: Stream ONLY the matched split file (~0.15s)
    results = []
    try:
        for fname in matching_files:
            file_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/{fname}"
            with fs.open(file_path, "rb") as f:
                pq_file = pq.ParquetFile(f)
                selected_cols = [c for col in COLUMNS_LIST if (c := col) in pq_file.schema.names]
                
                for rg_idx in range(pq_file.num_row_groups):
                    tbl = pq_file.read_row_group(rg_idx, columns=selected_cols)
                    
                    if "mobile" in tbl.column_names:
                        mobile_col = pc.cast(tbl["mobile"], pa.string()) if tbl["mobile"].type != pa.string() else tbl["mobile"]
                        mask = pc.equal(mobile_col, clean_mobile)
                        filtered = tbl.filter(mask)
                        
                        if filtered.num_rows > 0:
                            results.extend(filtered.to_pylist())
                            if len(results) >= limit:
                                break
                            
            if len(results) >= limit:
                break

        gc.collect()

        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": len(results),
            "data": results[:limit]
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

# 💓 3. HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "splits_indexed": len(master_index)
    }
