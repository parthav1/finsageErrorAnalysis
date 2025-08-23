#!/usr/bin/env python3
import sqlite3
import os
import sys
import json
import datetime

def create_frequent_qa_database(db_path):
    """
    Create a SQLite database for storing frequent QA pairs with proper schema
    
    Args:
        db_path (str): Path where the database file should be created
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (creates it if it doesn't exist)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the frequent_qa_pairs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS frequent_qa_pairs (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            question_rewritten TEXT NOT NULL,           
            answer TEXT,
            category TEXT NOT NULL,
            last_updated TIMESTAMP,
            updated_by TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            view_count INTEGER DEFAULT 0,
            version INTEGER DEFAULT 1,
            tags TEXT,
            metadata TEXT  -- SQLite doesn't have JSONB, using TEXT for JSON storage
        )
        ''')
        
        # Create the answer_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS answer_history (
            history_id INTEGER PRIMARY KEY,
            qa_id INTEGER,
            previous_answer TEXT,
            updated_answer TEXT,
            updated_at TIMESTAMP,
            updated_by TEXT,
            version INTEGER,
            FOREIGN KEY (qa_id) REFERENCES frequent_qa_pairs(id)
        )
        ''')
        
        # Create the feedback_question_aliases table (expanded version of question_aliases)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_question_aliases (
            alias_id INTEGER PRIMARY KEY,
            qa_id INTEGER,
            alias_text TEXT,
            session_id TEXT,
            response_id TEXT,
            rating INTEGER,
            question TEXT NOT NULL,
            question_rewritten TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'non_rag',
            is_match BOOLEAN DEFAULT TRUE,
            match_confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_by TEXT,
            approved_at TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (qa_id) REFERENCES frequent_qa_pairs(id)
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON frequent_qa_pairs (category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qa_id ON answer_history (qa_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alias_qa_id ON feedback_question_aliases (qa_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_response_id ON feedback_question_aliases (response_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON feedback_question_aliases (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alias_text ON feedback_question_aliases (alias_text)')
        
        # Set a busy timeout to wait for locks to clear
        cursor.execute('PRAGMA busy_timeout=5000')
        
        conn.commit()
        print(f"Database created successfully at {db_path}")
        print("Table structure:")
        
        for table in ['frequent_qa_pairs', 'answer_history', 'feedback_question_aliases']:
            print(f"\nTable: {table}")
            cursor.execute(f"PRAGMA table_info({table})")
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

def load_questions_into_database(db_path, json_file_path):
    """
    Load classified questions from JSON file into the database,
    adapting to the new merged_categorized.json structure which contains
    question objects with question, rewritten, and answer fields
    
    Args:
        db_path (str): Path to the SQLite database
        json_file_path (str): Path to the JSON file containing classified questions
    """
    try:
        # Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        current_time = datetime.datetime.now().isoformat()
        
        # Track unique questions to avoid duplicates
        unique_questions = set()
        duplicate_count = 0
        question_count = 0
        
        for category, category_data in data['categories'].items():
            for question_item in category_data['questions']:
                if isinstance(question_item, str):
                    question_text = question_item
                    rewritten_text = question_item 
                    answer_text = None 
                else:
                    question_text = question_item['question']
                    rewritten_text = question_item.get('rewritten', 'N/A') or 'N/A'
                    answer_text = question_item.get('answer','N/A') or 'N/A'
                
                # Skip if this is a duplicate question
                if question_text in unique_questions:
                    duplicate_count += 1
                    continue
                
                unique_questions.add(question_text)
                cursor.execute('''
                INSERT INTO frequent_qa_pairs 
                (question, question_rewritten, answer, category, last_updated, updated_by, metadata) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    question_text,
                    rewritten_text,
                    answer_text,
                    category, 
                    current_time,
                    'system',  # Initial import by system
                    json.dumps({'source': 'initial_import', 'import_date': current_time})
                ))
                question_count += 1
        
        conn.commit()
        print(f"Successfully loaded {question_count} unique questions into the database")
        if duplicate_count > 0:
            print(f"Skipped {duplicate_count} duplicate questions")
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        return False
    except sqlite3.Error as e:
        print(f"Error inserting data into database: {e}", file=sys.stderr)
        return False
    finally:
        if conn:
            conn.close()
    
    return True

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, '..', 'log','frequent_qa.db')
    
    if create_frequent_qa_database(db_path):
        print("Database schema created successfully.")
        
        # Load questions if JSON file is provided
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            json_path = sys.argv[1]
            if load_questions_into_database(db_path, json_path):
                print(f"Questions from {json_path} loaded successfully.")
            else:
                print(f"Failed to load questions from {json_path}.")
        else:
            print("No JSON file specified or file doesn't exist. Run with: python create_frequent_qa_database.py path/to/classified_questions.json")
    else:
        print("Failed to create database schema.")
