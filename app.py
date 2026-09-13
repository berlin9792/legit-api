import csv
import os
import re
import sqlite3
from flask import Flask, jsonify, render_template_string, request
from huggingface_hub import hf_hub_download

app = Flask(__name__)

DATASET_REPO = "Zerotracelegit/paytm"
FILENAME = "users.csv"
DB_FILE = "paytm_index.db"

BANK_KEYWORDS = [
    "bank",
    "sbi",
    "hdfc",
    "icici",
    "axis",
    "kotak",
    "pnb",
    "bob",
    "canara",
    "union",
    "indusind",
    "yes",
    "idfc",
    "paytm",
    "airtel",
    "rbl",
    "federal",
    "baroda",
    "uco",
    "gramin",
]


def clean_number(num):
    if not num:
        return ""
    cleaned = re.sub(r"[^0-9]", "", str(num))
    if len(cleaned) == 12 and cleaned.startswith("91"):
        cleaned = cleaned[2:]
    if len(cleaned) == 11 and cleaned.startswith("0"):
        cleaned = cleaned[1:]
    return cleaned[-10:] if len(cleaned) >= 10 else cleaned


# Cloud server par automatic background indexing
def build_smart_index():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, mobile TEXT, email TEXT, city TEXT,
            gender TEXT, address TEXT, dob TEXT, bank TEXT, state TEXT
        )
    """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_mobile ON users(mobile);")

    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] > 0:
        con.close()
        return

    print("⏳ Downloading users.csv on Cloud...")
    local_csv = hf_hub_download(
        repo_id=DATASET_REPO, filename=FILENAME, repo_type="dataset"
    )

    print("⚡ Indexing records...")
    with open(local_csv, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        batch = []

        for row in reader:
            if not row or len(row) < 2:
                continue

            name = row[0].strip() if len(row) > 0 else ""
            mobile = ""
            email = ""
            bank = ""
            gender = ""
            city = ""
            state = ""
            address = ""
            dob = ""

            for cell in row:
                val = str(cell).strip()
                val_lower = val.lower()
                if not val:
                    continue

                if not mobile:
                    clean_m = clean_number(val)
                    if (
                        len(clean_m) == 10
                        and clean_m.isdigit()
                        and clean_m[0] in "6789"
                    ):
                        mobile = clean_m
                        continue

                if "@" in val and "." in val and not email:
                    email = val
                    continue

                if (
                    val_lower in ["male", "female", "m", "f", "transgender"]
                    and not gender
                ):
                    gender = (
                        "Male"
                        if val_lower in ["male", "m"]
                        else "Female"
                        if val_lower in ["female", "f"]
                        else val
                    )
                    continue

                if not bank:
                    for b_key in BANK_KEYWORDS:
                        if b_key in val_lower and not any(
                            x in val_lower
                            for x in [".com", "@", "road", "street"]
                        ):
                            bank = val
                            break

            def get_c(idx):
                return row[idx].strip() if idx < len(row) and row[idx] else ""

            if not city:
                city = get_c(3)
            if not gender:
                gender = get_c(4)
            if not address:
                address = get_c(5)
            if not dob:
                dob = get_c(6)
            if not bank:
                bank = get_c(7)
            if not state:
                state = get_c(8)

            if not mobile and not name:
                continue

            batch.append(
                (name, mobile, email, city, gender, address, dob, bank, state)
            )

            # 25k batch for low RAM safety
            if len(batch) >= 25000:
                cur.executemany(
                    "INSERT INTO users (name, mobile, email, city, gender, address, dob, bank, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                con.commit()
                batch = []

        if batch:
            cur.executemany(
                "INSERT INTO users (name, mobile, email, city, gender, address, dob, bank, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            con.commit()

    # Index banne ke baad 1.29 GB raw csv delete kar do disk bachane ke liye
    if os.path.exists(local_csv):
        os.remove(local_csv)

    cur.execute("SELECT COUNT(*) FROM users")
    total = cur.fetchone()[0]
    con.close()
    print(f"✅ Index Complete! Total Records: {total:,}")


# Startup Build
build_smart_index()


# API Endpoint
@app.route("/api/paytm")
def paytm_api():
    raw_num = request.args.get("number", "").strip()
    mobile = clean_number(raw_num)

    if not mobile:
        return (
            jsonify({"status": False, "message": "Provide ?number=9502202203"}),
            400,
        )

    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT * FROM users WHERE mobile = ? LIMIT 10", (mobile,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    if not rows:
        return (
            jsonify(
                {
                    "status": False,
                    "number": mobile,
                    "message": "Number not found in database",
                }
            ),
            404,
        )

    cleaned = []
    for r in rows:
        item = {
            "name": r.get("name") or "N/A",
            "mobile": r.get("mobile") or "N/A",
            "email": r.get("email") or "N/A",
            "bank": r.get("bank") or "N/A",
            "city": r.get("city") or "N/A",
            "state": r.get("state") or "N/A",
            "gender": r.get("gender") or "N/A",
            "address": r.get("address") or "N/A",
            "dob": r.get("dob") or "N/A",
        }
        cleaned.append({k: v for k, v in item.items() if v != "N/A"})

    return jsonify(
        {
            "status": True,
            "number": mobile,
            "total_matches": len(cleaned),
            "data": cleaned if len(cleaned) > 1 else cleaned[0],
        }
    )


# Web Search UI
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Paytm Search API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin:0; padding:0; box-sizing:border-box; font-family: 'Segoe UI', Tahoma, sans-serif; }
            body { background: #0b132b; color: #fff; max-width: 700px; margin: 40px auto; padding: 20px; }
            h2 { color: #48cae4; text-align: center; margin-bottom: 5px; }
            p.sub { color: #8d99ae; text-align: center; font-size: 14px; margin-bottom: 25px; }
            .box { display: flex; gap: 10px; margin-bottom: 20px; }
            input { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #1c2541; background: #1c2541; color: #fff; font-size: 16px; outline: none; }
            input:focus { border-color: #48cae4; }
            button { padding: 14px 24px; background: #48cae4; color: #000; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; }
            pre { background: #1c2541; padding: 15px; border-radius: 8px; color: #caf0f8; font-size: 14px; overflow-x: auto; margin-top: 15px; }
        </style>
    </head>
    <body>
        <h2>⚡ Paytm Database Fast Lookup</h2>
        <p class="sub">24/7 Live Cloud API</p>
        <div class="box">
            <input id="num" placeholder="Enter Mobile Number" onkeypress="if(event.key==='Enter') search()">
            <button onclick="search()">Search</button>
        </div>
        <pre id="output" style="display:none;"></pre>
        <script>
            async function search() {
                const n = document.getElementById('num').value.trim();
                const out = document.getElementById('output');
                if(!n) return alert('Number daalo!');
                out.style.display = 'block';
                out.textContent = 'Searching database...';
                try {
                    const r = await fetch('/api/paytm?number=' + encodeURIComponent(n));
                    const j = await r.json();
                    out.textContent = JSON.stringify(j, null, 2);
                } catch(e) {
                    out.textContent = 'Error: ' + e;
                }
            }
        </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
