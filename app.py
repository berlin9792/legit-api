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
# ⚙️ CONFIGURATION (100% PUBLIC BUCKET ACCESS)
# ==============================================================================
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required Columns to return
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
# ==============================================================================

fs = None
master_index = []

def fetch_master_index():
    """Hugging Face Bucket se consolidated index.json load karta hai"""
    global fs, master_index
    try:
        if fs is None:
            fs = HfFileSystem()
            
        index_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/index.json"
        
        if fs.exists(index_path):
            with fs.open(index_path, "rb") as f:
                master_index = json.loads(f.read().decode())
            print(f"✅ Master Index loaded successfully! ({len(master_index)} splits indexed)")
            return True
        else:
            print("⚠️ index.json not found in bucket.")
            return False
    except Exception as e:
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
    version="4.6",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🏠 1. HOME ENDPOINT
@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "Master Indexed Binary Search",
        "total_indexed_files": len(master_index),
        "index_status": "Ready 🟢" if len(master_index) > 0 else "Not Ready ❌",
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
                status_code=503, 
                detail="Dataset index is not available. Please check bucket."
            )

    # ⚡ STEP 1: Find matching files using In-Memory Index (Takes 0.0001 sec)
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

    # ⚡ STEP 2: Read ONLY the exact matching file(s) (Takes 0.1s - 0.2s)
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
                        # ✅ FIXED: pa.string() instead of pc.string()
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# 💓 3. HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "splits_loaded": len(master_index)
    }
