# Stage 8: Database Analytics Code for Storing OCR Results and Generating Analytics Reports

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Connect to the SQLite database (or create it if it doesn't exist)
db_path = 'ocr_results.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table for storing OCR results if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS ocr_results (
    id INTEGER PRIMARY KEY,
    result TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Function to insert OCR results into the database

def insert_ocr_result(result):
    cursor.execute('INSERT INTO ocr_results (result) VALUES (?)', (result,))
    conn.commit()

# Function to generate analytics report

def generate_analytics_report():
    # Query data from the database
    cursor.execute('SELECT * FROM ocr_results')
    data = cursor.fetchall()

    # Create a DataFrame for analysis
    df = pd.DataFrame(data, columns=['id', 'result', 'timestamp'])

    # Example analysis: Count results over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    report = df.resample('D').count()

    # Plotting the analytics report
    plt.figure(figsize=(10, 6))
    plt.plot(report.index, report['result'], marker='o')
    plt.title('Daily Count of OCR Results')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analytics_report.png')
    plt.close()

# Sample usage of the functions
insert_ocr_result('Sample OCR Result')  # Replace with the actual result
generate_analytics_report()

# Close the database connection
conn.close()