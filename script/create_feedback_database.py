#!/usr/bin/env python3
import sqlite3
import os
import sys

def create_response_log_database(db_path):
    """
    Create a SQLite database for storing response logs with proper schema
    
    Args:
        db_path (str): Path where the database file should be created
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (creates it if it doesn't exist)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the feedback table with appropriate columns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            response_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            user TEXT,
            feedback TEXT,
            question TEXT,
            response TEXT,
            is_rag INTEGER DEFAULT 1 NOT NULL,
            log TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON feedback (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_response_id ON feedback (response_id)')
        
        # Set a busy timeout to wait for locks to clear
        cursor.execute('PRAGMA busy_timeout=5000')
        
        conn.commit()
        print(f"Database created successfully at {db_path}")
        print("Table structure:")
        cursor.execute("PRAGMA table_info(feedback)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]}){' PRIMARY KEY' if col[5] == 1 else ''}")
            
    except sqlite3.Error as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        return False
    finally:
        if conn:
            conn.close()
            
    return True

if __name__ == "__main__":
    # Default path or allow specifying via command line
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default path: create in a 'log' directory relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, '..', 'log', 'feedback.db')
    
    create_response_log_database(db_path)
