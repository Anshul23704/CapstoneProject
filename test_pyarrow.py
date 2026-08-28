import sqlite3
import pandas as pd
import pyarrow as pa
from pyarrow.pandas_compat import dataframe_to_arrays

p = "/Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject/output/20260827_143815/results.db"
con = sqlite3.connect(p)
db = pd.read_sql_query("SELECT * FROM ocr_results", con)
con.close()

# Simulate what GUI does
for col in db.select_dtypes(include=['object']).columns:
    db[col] = db[col].astype(str)

try:
    pa.Table.from_pandas(db)
    print("Pyarrow Table creation successful")
except Exception as e:
    print(f"Failed Table: {type(e).__name__}: {e}")

