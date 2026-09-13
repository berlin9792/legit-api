import os
import re
from flask import Flask, jsonify, render_template_string, request
import requests

app = Flask(__name__)

DATASET_REPO = "Zerotracelegit/paytm"

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


# ============================================================
# API ENDPOINT: /api/paytm?number=9502202203
# ============================================================
@app.route("/api/paytm")
def paytm_api():
    raw_num = request.args.get("number", "").strip()
    mobile = clean_number(raw_num)

    if not mobile or len(mobile) < 10:
        return (
            jsonify(
                {
                    "status": False,
                    "message": "Valid 10-digit mobile number provide karein. Example: /api/paytm?number=9502202203",
                }
            ),
            400,
        )

    try:
        # Hugging Face Cloud Search Engine (Instant Serverless API)
        hf_url = "https://datasets-server.huggingface.co/search"
        params = {
            "dataset": DATASET_REPO,
            "config": "default",
            "split": "train",
            "query": mobile,
            "limit": 10,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        res = requests.get(hf_url, params=params, headers=headers, timeout=9)

        if res.status_code != 200:
            return (
                jsonify(
                    {
                        "status": False,
                        "number": mobile,
                        "message": "Dataset server busy ya record nahi mila",
                    }
                ),
                404,
            )

        data = res.json()
        rows = data.get("rows", [])

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

        matched_records = []

        for r in rows:
            row_dict = r.get("row", {})

            # Saari values extract karke smart detect karo
            name = row_dict.get("name") or row_dict.get("column0") or ""
            row_mob = ""
            email = ""
            bank = ""
            gender = ""
            city = row_dict.get("city") or row_dict.get("column3") or ""
            state = row_dict.get("state") or row_dict.get("column8") or ""
            address = (
                row_dict.get("address") or row_dict.get("column5") or "N/A"
            )
            dob = row_dict.get("dob") or row_dict.get("column6") or "N/A"

            # Auto-Detect values from row
            for k, val in row_dict.items():
                val_str = str(val).strip()
                val_lower = val_str.lower()

                if not val_str or val_str.lower() in ["nan", "null", "none"]:
                    continue

                # Mobile
                if not row_mob:
                    cm = clean_number(val_str)
                    if len(cm) == 10 and cm.isdigit() and cm[0] in "6789":
                        row_mob = cm

                # Email
                if "@" in val_str and "." in val_str and not email:
                    email = val_str

                # Gender
                if (
                    val_lower in ["male", "female", "m", "f", "transgender"]
                    and not gender
                ):
                    gender = (
                        "Male"
                        if val_lower in ["male", "m"]
                        else "Female"
                        if val_lower in ["female", "f"]
                        else val_str
                    )

                # Bank
                if not bank:
                    for b_key in BANK_KEYWORDS:
                        if b_key in val_lower and not any(
                            x in val_lower
                            for x in [".com", "@", "road", "street"]
                        ):
                            bank = val_str
                            break

            # Agar number match karta hai
            if row_mob == mobile or mobile in str(row_dict):
                item = {
                    "name": name if name else "N/A",
                    "mobile": row_mob if row_mob else mobile,
                    "email": email if email else "N/A",
                    "bank": bank if bank else "N/A",
                    "city": city if city else "N/A",
                    "state": state if state else "N/A",
                    "gender": gender if gender else "N/A",
                    "address": address,
                    "dob": dob,
                }
                # Filter out N/A fields for cleaner JSON
                matched_records.append(
                    {k: v for k, v in item.items() if v != "N/A"}
                )

        if not matched_records:
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

        return jsonify(
            {
                "status": True,
                "number": mobile,
                "total_matches": len(matched_records),
                "data": matched_records
                if len(matched_records) > 1
                else matched_records[0],
            }
        )

    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500


# ============================================================
# WEB UI
# ============================================================
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Paytm API Lookup - Vercel</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin:0; padding:0; box-sizing:border-box; font-family: 'Segoe UI', Tahoma, sans-serif; }
            body { background: #0b132b; color: #fff; max-width: 700px; margin: 40px auto; padding: 20px; }
            h2 { color: #48cae4; text-align: center; margin-bottom: 5px; }
            p.sub { text-align: center; color: #8d99ae; font-size: 14px; margin-bottom: 25px; }
            .box { display: flex; gap: 10px; margin-bottom: 20px; }
            input { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #1c2541; background: #1c2541; color: #fff; font-size: 16px; outline: none; }
            input:focus { border-color: #48cae4; }
            button { padding: 14px 24px; background: #48cae4; color: #000; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; }
            pre { background: #1c2541; padding: 15px; border-radius: 8px; color: #caf0f8; font-size: 14px; overflow-x: auto; margin-top: 15px; }
        </style>
    </head>
    <body>
        <h2>⚡ Paytm Database Lookup (Vercel Live)</h2>
        <p class="sub">High-Speed Serverless API</p>
        <div class="box">
            <input id="num" placeholder="Enter Mobile (e.g. 9502202203)" onkeypress="if(event.key==='Enter') search()">
            <button onclick="search()">Search</button>
        </div>
        <pre id="output" style="display:none;"></pre>
        <script>
            async function search() {
                const n = document.getElementById('num').value.trim();
                const out = document.getElementById('output');
                if(!n) return alert('Number daalo!');
                out.style.display = 'block';
                out.textContent = 'Searching...';
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


app = app
