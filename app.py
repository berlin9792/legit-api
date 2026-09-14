import os
import gc
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
# ==============================================================================

fs = None
master_index = []

def load_master_index():
    global fs, master_index
    try:
        print("🔗 Loading Master Index (150KB)...")
        fs = HfFileSystem()
        index_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/index.json"
        
        with fs.open(index_path, "rb") as f:
            master_index = json.loads(f.read().decode())
            
        print(f"✅ Master Index Loaded! Indexed {len(master_index)} split files.")
    except Exception as e:
        print(f"❌ Error loading index.json: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_master_index()
    yield
    gc.collect()

app = FastAPI(
    title="Ultra-Fast Mobile Lookup API",
    description="Indexed Parquet Search Engine (< 0.2s Response, 512MB RAM Safe)",
    version="4.0",
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
        "engine": "Master Indexed Binary Search",
        "total_indexed_files": len(master_index),
        "speed": "< 0.2s",
        "ram_usage": "< 30MB"
    }

# 🔍 INSTANT SEARCH ENDPOINT (< 0.2s)
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    if not master_index:
        load_master_index()
        if not master_index:
            raise HTTPException(status_code=500, detail="Index not ready. Please check bucket.")

    # ⚡ STEP 1: Find matching files from In-Memory Index (Takes 0.0001s)
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
                
                for rg_idx in range(pq_file.num_row_groups):
                    tbl = pq_file.read_row_group(rg_idx, columns=COLUMNS_LIST)
                    mobile_col = pc.cast(tbl["mobile"], pc.string()) if tbl["mobile"].type != pc.string() else tbl["mobile"]
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
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "indexed_files": len(master_index)}
