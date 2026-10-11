import sqlite3
import os

DB_PATH = "data/attendance.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table 1: Daily Summary
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            name TEXT,
            date TEXT,
            check_in_time TEXT,
            check_out_time TEXT,
            status TEXT DEFAULT 'Present'
        )
    ''')
    
    # Table 2: Raw Audit Log (Every event)
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            camera TEXT,
            action TEXT,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ SQLite Database initialized at data/attendance.db")

if __name__ == "__main__":
    init_db()