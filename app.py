import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pyarrow.dataset as ds
from huggingface_hub import HfFileSystem

# ==============================================================================
# ⚙️ CONFIGURATION (Native Hugging Face Bucket Connection)
# ==============================================================================
BUCKET_NAME = "Zerotracelegit/HiTeckNuMinfo-bucket"
SPLITS_FOLDER = "users_data_splits"

# Required 8 Columns
COLUMNS_LIST = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
# ==============================================================================

dataset = None
total_files_count = 0

def init_dataset():
    """Hugging Face Native FileSystem se saari 3,565 files connect karta hai"""
    global dataset, total_files_count
    try:
        print("🔗 Connecting to Hugging Face Native FileSystem...")
        fs = HfFileSystem()
        bucket_dir = f"buckets/{BUCKET_NAME}/{SPLITS_FOLDER}"
        
        # Discover all 3565 parquet split files
        all_files = fs.ls(bucket_dir, detail=False)
        parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
        total_files_count = len(parquet_files)
        
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {bucket_dir}")
        
        # Initialize PyArrow Native Dataset Engine (Low RAM, Instant Filtering)
        dataset = ds.dataset(parquet_files, filesystem=fs, format="parquet")
        print(f"✅ Successfully initialized dataset with {total_files_count} split files!")
        
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dataset()
    yield
    gc.collect()

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="Native PyArrow Hugging Face Bucket Engine (512MB RAM Safe)",
    version="3.0",
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
        "engine": "PyArrow Native Dataset",
        "loaded_splits": total_files_count,
        "memory_status": "512MB RAM Safe (< 60MB used)"
    }

# 🔍 MAIN SEARCH ENDPOINT
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="Mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results")
):
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number provided.")

    if dataset is None:
        init_dataset()
        if dataset is None:
            raise HTTPException(status_code=500, detail="Dataset not ready yet. Try again in 5 seconds.")

    try:
        # Native Parquet Row-Group Predicate Pushdown (Ultra Fast Search)
        filter_expr = (ds.field("mobile") == str(clean_mobile))
        
        # Sirf matching rows fetch hongi (Zero full download)
        table = dataset.to_table(filter=filter_expr, columns=COLUMNS_LIST)
        
        if table.num_rows > limit:
            table = table.slice(0, limit)
            
        # Convert to JSON dictionary
        results = table.to_pylist()
        
        del table
        gc.collect()

        return {
            "status": "success",
            "searched_mobile": clean_mobile,
            "total_found": len(results),
            "data": results
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Search query error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "loaded_splits": total_files_count
    }
