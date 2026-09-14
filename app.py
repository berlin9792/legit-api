import os
import gc
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb

app = FastAPI(
    title="High-Speed Mobile Lookup API",
    description="Ultra-fast Parquet search engine optimized for Render 512MB RAM",
    version="2.0"
)

# CORS Enabled (Taaki website/frontend se bhi direct call kar sako)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌐 HUGGING FACE BUCKET SPLITS PATH (Wildcard *.parquet se saare parts connect honge)
PARQUET_GLOB_URL = "https://huggingface.co/buckets/Zerotracelegit/HiTeckNuMinfo-bucket/resolve/main/users_data_splits/*.parquet"

# Agar Bucket private hai toh Render Environment me HF_TOKEN set kar sakte hain
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- 🛡️ DUCKDB 512MB STRICT RAM LOCK CONFIGURATION ---
con = duckdb.connect(database=":memory:", read_only=False)
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET memory_limit='200MB';")          # 200MB strict cap (Render safe)
con.execute("SET threads=1;")                     # 1 thread = Zero RAM overhead
con.execute("SET enable_object_cache=false;")     # Cache off (Saves RAM)
con.execute("SET preserve_insertion_order=false;")

if HF_TOKEN:
    con.execute(f"SET http_headers = '{{\"Authorization\": \"Bearer {HF_TOKEN}\"}}';")

# Required 8 Columns
COLUMNS = "mobile, name, fname, address, alt, circle, id, email"


# 🏠 1. HOME ENDPOINT
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Mobile Lookup API",
        "storage": "Hugging Face Bucket (Sorted Parquet Chunks)",
        "memory_safety": "Render 512MB RAM Compliant"
    }


# 🔍 2. MAIN SEARCH ENDPOINT (Fetch by Mobile Number)
@app.get("/search")
def search_mobile(
    mobile: str = Query(..., description="10-digit mobile number to search", min_length=5, max_length=15),
    limit: int = Query(5, le=10, description="Max results (default 5)")
):
    # Input Sanitization (Sirf digits rakho)
    clean_mobile = "".join(filter(str.isdigit, mobile.strip()))
    if not clean_mobile:
        raise HTTPException(status_code=400, detail="Invalid mobile number. Only digits allowed.")

    try:
        cursor = con.cursor()

        # Ultra-Fast Query: Sorted data hone ki wajah se sirf 1-2 row groups check honge
        query = f"""
            SELECT {COLUMNS}
            FROM read_parquet('{PARQUET_GLOB_URL}')
            WHERE CAST(mobile AS VARCHAR) = ?
            LIMIT {limit}
        """
        
        cursor.execute(query, [clean_mobile])
        
        # Zero-Pandas Native Fetch (Instant & 0% Memory Spike)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        # Format rows into clean JSON
        results = [dict(zip(col_names, row)) for row in rows]

        # Explicit Garbage Collection (Har request ke baad RAM free)
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


# 💓 3. HEALTH CHECK ENDPOINT (Render ko 24/7 Jagaye Rakhne ke liye)
@app.get("/health")
def health():
    return {"status": "ok", "ram_state": "healthy"}
