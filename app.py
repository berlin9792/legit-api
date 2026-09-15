import os
import gc
import json
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_SEARCH_WORKERS = 16  # 16 Parallel threads for instant lookup
# ==============================================================================

fs = HfFileSystem(token=HF_TOKEN, default_block_size=16 * 1024 * 1024) if HF_TOKEN else HfFileSystem(default_block_size=16 * 1024 * 1024)
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
        print(f"✅ Master Index loaded! ({len(master_index)} splits)")
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
    description="Sub-Second Parallel Parquet Search Engine",
    version="9.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🏠 1. HOME STATUS
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    if not master_index:
        fetch_master_index()
        
    return {
        "status": "online",
        "engine": "16X Parallel Vector Search",
        "total_indexed_files": len(master_index),
        "index_status": "Ready 🟢" if len(master_index) > 0 else "Loading Error ❌",
        "sample_url": "/sample",
        "memory_limit": "512MB Safe (< 50MB Active RAM)"
    }

# 🎁 2. SAMPLE NUMBERS ENDPOINT
@app.get("/sample")
def get_sample_numbers():
    if not master_index:
        fetch_master_index()
        if not master_index:
            raise HTTPException(status_code=500, detail="Index not ready.")

    try:
        sample_file = master_index[len(master_index)//2]["file"]
        file_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/{sample_file}"
        
        with fs.open(file_path, "rb") as f:
            pq_file = pq.ParquetFile(f)
            selected_cols = [c for col in COLUMNS_LIST if (c := col) in pq_file.schema.names]
            tbl = pq_file.read_row_group(0, columns=selected_cols).slice(0, 5)
            samples = tbl.to_pylist()
            
        return {
            "status": "success",
            "message": "Real records from dataset:",
            "sample_records": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample error: {str(e)}")

def scan_file_for_number(fname: str, candidates: list, stop_event: threading.Event, limit: int):
    """Single file fast scanner with instant stop signal"""
    if stop_event.is_set():
        return []
        
    file_path = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}/{fname}"
    try:
        with fs.open(file_path, "rb") as f:
            pq_file = pq.ParquetFile(f)
            meta = pq_file.metadata
            
            if "mobile" not in pq_file.schema.names:
                return []
                
            col_idx = pq_file.schema.names.index("mobile")
            selected_cols = [c for col in COLUMNS_LIST if (c := col) in pq_file.schema.names]
            
            # Check row groups min/max stats
            matched_rgs = []
            for rg_idx in range(meta.num_row_groups):
                stat = meta.row_group(rg_idx).column(col_idx).statistics
                if stat and stat.has_min_max:
                    rg_min = str(stat.min)
                    rg_max = str(stat.max)
                    if any(rg_min <= cand <= rg_max for cand in candidates):
                        matched_rgs.append(rg_idx)
                else:
                    matched_rgs.append(rg_idx)

            if not matched_rgs:
                return []

            # Read matched row groups
            records = []
            for rg_idx in matched_rgs:
                if stop_event.is_set():
                    break
                    
                tbl = pq_file.read_row_group(rg_idx, columns=selected_cols)
                mobile_col = pc.cast(tbl["mobile"], pa.string()) if tbl["mobile"].type != pa.string() else tbl["mobile"]
                
                mask = pc.is_in(mobile_col, value_set=pa.array(candidates))
                filtered = tbl.filter(mask)
                
                if filtered.num_rows > 0:
                    records.extend(filtered.to_pylist())
                    stop_event.set() # Instant stop signal to all other 15 threads!
                    return records[:limit]

            return records
    except Exception:
        return []

# 🔍 3. 16X PARALLEL SUB-SECOND SEARCH ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    global master_index
    raw_digits = "".join(filter(str.isdigit, mobile.strip()))
    if not raw_digits:
        raise HTTPException(status_code=400, detail="Invalid mobile number.")

    if not master_index:
        fetch_master_index()
        if not master_index:
            raise HTTPException(status_code=500, detail=f"Index error: {load_error}")

    # Variations (10-digit, 91 prefix, 0 prefix)
    search_candidates = {raw_digits}
    if len(raw_digits) == 10:
        search_candidates.add("91" + raw_digits)
        search_candidates.add("0" + raw_digits)
    elif len(raw_digits) == 12 and raw_digits.startswith("91"):
        search_candidates.add(raw_digits[2:])
    elif len(raw_digits) == 11 and raw_digits.startswith("0"):
        search_candidates.add(raw_digits[1:])

    candidate_list = list(search_candidates)

    # Step 1: In-memory index matching
    matching_files = set()
    for candidate in candidate_list:
        for item in master_index:
            if item["min"] <= candidate <= item["max"]:
                matching_files.add(item["file"])

    if not matching_files:
        return {
            "status": "success",
            "searched_mobile": raw_digits,
            "total_found": 0,
            "data": []
        }

    # Step 2: 16 Parallel Workers Execution with Early Stop
    all_results = []
    stop_event = threading.Event()

    try:
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            futures = [
                executor.submit(scan_file_for_number, fname, candidate_list, stop_event, limit)
                for fname in matching_files
            ]
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_results.extend(res)
                    if len(all_results) >= limit:
                        stop_event.set()
                        break

        gc.collect()

        return {
            "status": "success",
            "searched_mobile": raw_digits,
            "total_found": len(all_results),
            "data": all_results[:limit]
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Search execution error: {str(e)}")

# 💓 4. HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "ok", "splits_indexed": len(master_index)}
