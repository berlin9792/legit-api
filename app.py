import os
import gc
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Required 8 Columns
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
MAX_SEARCH_WORKERS = 8  # 8 Parallel threads (512MB RAM Safe)
# ==============================================================================

fs = None
parquet_files_list = []

def init_system():
    global fs, parquet_files_list
    try:
        print("🔗 Connecting to Hugging Face Native Storage...")
        fs = HfFileSystem()
        bucket_dir = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}"
        
        all_files = fs.ls(bucket_dir, detail=False)
        parquet_files_list = sorted([f for f in all_files if f.endswith(".parquet")])
        print(f"✅ Successfully registered {len(parquet_files_list)} sorted split files!")
    except Exception as e:
        print(f"❌ Error during file listing: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system()
    yield
    gc.collect()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="Statistical Range-Skipping Parquet Engine (512MB RAM Safe)",
    version="3.5",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def scan_single_file(file_path: str, target_mobile: str, limit: int):
    """1KB Footer Metadata scan karke sirf matching file ko read karta hai"""
    try:
        with fs.open(file_path, "rb") as f:
            pq_file = pq.ParquetFile(f)
            meta = pq_file.metadata
            
            # Check if mobile column exists
            if "mobile" not in pq_file.schema.names:
                return []
                
            col_idx = pq_file.schema.names.index("mobile")
            matching_row_groups = []

            # Range-Check using Min/Max statistics (Instant Skip)
            for rg_idx in range(meta.num_row_groups):
                rg_meta = meta.row_group(rg_idx)
                col_stat = rg_meta.column(col_idx).statistics
                
                if col_stat and col_stat.has_min_max:
                    min_val = str(col_stat.min)
                    max_val = str(col_stat.max)
                    # Range check: Agar number is range me nahi hai toh SKIP
                    if min_val <= target_mobile <= max_val:
                        matching_row_groups.append(rg_idx)
                else:
                    matching_row_groups.append(rg_idx)

            # Agar number is file ke range me nahi hai, 0 bytes data download!
            if not matching_row_groups:
                return []

            # Number range me hai! Read only specific row groups & columns
            results = []
            for rg_idx in matching_row_groups:
                tbl = pq_file.read_row_group(rg_idx, columns=COLUMNS_LIST)
                
                # Fast Filter
                mobile_col_str = pc.cast(tbl["mobile"], pc.string()) if tbl["mobile"].type != pc.string() else tbl["mobile"]
                mask = pc.equal(mobile_col_str, target_mobile)
                filtered_tbl = tbl.filter(mask)
                
                if filtered_tbl.num_rows > 0:
                    results.extend(filtered_tbl.to_pylist())
                    if len(results) >= limit:
                        return results[:limit]

            return results
    except Exception:
        return []

@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "Statistical Range-Skipping Engine",
        "total_splits": len(parquet_files_list),
        "ram_safety": "Render 512MB RAM Safe (< 50MB Active RAM)"
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

    if not parquet_files_list:
        init_system()
        if not parquet_files_list:
            raise HTTPException(status_code=500, detail="Bucket files not ready yet.")

    all_matched_records = []

    # Parallel Footer Scanning (8 Workers = Zero RAM spikes)
    try:
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            futures = {
                executor.submit(scan_single_file, fpath, clean_mobile, limit): fpath 
                for fpath in parquet_files_list
            }
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_matched_records.extend(res)
                    # Early Exit: Agar limit poori ho gayi toh turant return karo!
                    if len(all_matched_records) >= limit:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

        gc.collect()

        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": len(all_matched_records),
            "data": all_matched_records[:limit]
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Lookup error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "splits_loaded": len(parquet_files_list)}
