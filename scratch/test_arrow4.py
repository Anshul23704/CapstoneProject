import pandas as pd
import sqlite3
import pyarrow as pa
import sys

db_path = '/Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject/output/20260827_143815/results.db'
con = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM ocr_results", con)
con.close()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

try:
    pa.Table.from_pandas(df)
    print("Arrow OK")
except Exception as e:
    print(f"Arrow Error: {e}")
