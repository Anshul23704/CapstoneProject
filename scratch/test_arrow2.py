import pandas as pd
import sqlite3
import pyarrow as pa
import sys
import glob
import os

db_files = glob.glob('/Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject/output/*/results.db')
if not db_files:
    print("No db files found")
    sys.exit(0)

for db_path in db_files:
    print(f"Testing {db_path}...")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM ocr_results", con)
    con.close()
    
    if df.empty:
        print("Empty")
        continue

    try:
        table = pa.Table.from_pandas(df)
        print("Arrow OK")
    except Exception as e:
        print(f"Arrow Error: {e}")
