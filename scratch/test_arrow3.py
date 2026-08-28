import pandas as pd
import pyarrow as pa
import os
import glob
import sys

files = glob.glob('/Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject/output/20260827_143815/*.csv')
for f in files:
    try:
        df = pd.read_csv(f)
        pa.Table.from_pandas(df)
        print(f"{os.path.basename(f)}: OK")
    except Exception as e:
        print(f"{os.path.basename(f)}: ERROR: {e}")

