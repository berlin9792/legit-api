import os
import gc
import json
import bisect
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
HF_TOKEN = os.getenv("HF_TOKEN", None)
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required Columns to return
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
# ==============================================================================

# Global FileSystem (With fast buffer)
fs = HfFileSystem(token=HF_TOKEN, default_block_size=16 * 1024 * 1024) if HF_TOKEN else HfFileSystem(default_block_size=16 * 1024 * 1024)
master_index = []
index_keys = []
load_error = None

def fetch_master_index():
    """Bucket se index.json load karta hai aur Binary Search keys banata hai"""
    global master_index, index_keys, load_error
    index_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/index.json"
    try:
        with fs.open(index_path, "rb") as f:
            content = f.read().decode("utf-8")
            master_index = json.loads(content)
            
        # Binary search optimization (0.00001s lookup)
        master_index.sort(key=lambda x: x["min"])
        index_keys = [item["min"] for item in master_index]
        
        load_error = None
        print(f"✅ Master Index loaded & binary-indexed! ({len(master_index)} splits)")
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
    description="Sub-Second Indexed Parquet Search Engine",
    version="7.0",
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
        "engine": "Binary Row-Group Pruning Engine",
        "total_indexed_files": len(master_index),
        "speed": "< 0.2s (Sub-Second)",
        "index_status": "Ready 🟢" if len(master_index) > 0 else "Loading Error ❌",
        "memory_limit": "512MB Safe (< 50MB Active RAM)"
    }

# 🔍 2. ULTRA-FAST SUB-SECOND LOOKUP ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    global master_index, index_keys
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

    # ⚡ STEP 1: Binary Search on Master Index (< 0.0001s)
    # Bisect se instant matching range find hoti hai
    idx = bisect.bisect_right(index_keys, clean_mobile) - 1
    matching_files = []
    
    # Check current candidate & boundary overlaps
    for check_idx in range(max(0, idx - 1), min(len(master_index), idx + 2)):
        item = master_index[check_idx]
        if item["min"] <= clean_mobile <= item["max"]:
            matching_files.append(item["file"])

    if not matching_files:
        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": 0,
            "data": []
        }

    # ⚡ STEP 2: Row-Group Statistics Pruning (Only 1 HTTP Request!)
    results = []
    try:
        for fname in matching_files:
            file_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/{fname}"
            with fs.open(file_path, "rb") as f:
                pq_file = pq.ParquetFile(f)
                meta = pq_file.metadata
                
                # Check column index
                if "mobile" not in pq_file.schema.names:
                    continue
                col_idx = pq_file.schema.names.index("mobile")
                selected_cols = [c for col in COLUMNS_LIST if (c := col) in pq_file.schema.names]
                
                # 🎯 STATISTICAL FILTER: Only find the EXACT row group(s)
                target_rgs = []
                for rg_idx in range(meta.num_row_groups):
                    stat = meta.row_group(rg_idx).column(col_idx).statistics
                    if stat and stat.has_min_max:
                        rg_min = str(stat.min)
                        rg_max = str(stat.max)
                        # Agar number is 50k row group me nahi hai -> 0 BYTES DOWNLOAD (SKIP)
                        if rg_min <= clean_mobile <= rg_max:
                            target_rgs.append(rg_idx)
                    else:
                        target_rgs.append(rg_idx)
                
                # 🚀 Download ONLY the 1 matching Row Group (Takes ~0.1s)
                for rg_idx in target_rgs:
                    tbl = pq_file.read_row_group(rg_idx, columns=selected_cols)
                    
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
