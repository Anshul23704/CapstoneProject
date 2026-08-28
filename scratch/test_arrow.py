import pandas as pd
import pyarrow as pa
import sqlite3

con = sqlite3.connect('/Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject/output/20260827_143815/results.db')
df = pd.read_sql_query("SELECT * FROM ocr_results", con)
con.close()

try:
    table = pa.Table.from_pandas(df)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
