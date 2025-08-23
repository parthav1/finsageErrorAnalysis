#!/usr/bin/env python3
import sqlite3
import json
import time
import datetime
import logging
import os
from openai import OpenAI
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedback_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feedback_processor")

FEEDBACK_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'feedback.db')
FREQUENT_QA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frequent_qa.db')
LAST_PROCESSED_ID_FILE = "last_processed_id.txt"
PROCESSING_INTERVAL_SECONDS = 600  
API_KEY = "sk-BUYcNLayB5w9rHwa1gYodJq3bgbJX6ChWNeRBNe1I6CgX4zr"

client = OpenAI(api_key=API_KEY,base_url="https://api.lkeap.cloud.tencent.com/v1")

CATEGORIES = [
    "Company_Basics_Governance",
    "Financial_Performance_Metrics",
    "Sales_Market_Performance",
    "Products_Technology",
    "Strategy_Development",
    "Major_Transactions_Agreements"
]

# Detailed descriptions of question categories
CATEGORY_DESCRIPTIONS = {
    "Company_Basics_Governance": "Company structure, headquarters, VIE, relationships with other companies, shareholders, board members, executives, employees, locations, history",
    "Financial_Performance_Metrics": "Stock information, ADS, revenue, profit margins, cash flow, financial results, dividends, R&D investment, financing",
    "Sales_Market_Performance": "Stores, regional sales, sales data, product pricing, market expansion, sales channels, customer segments",
    "Products_Technology": "Product lines, delivery timelines, technological advantages, R&D capabilities, driving systems, charging, manufacturing, suppliers",
    "Strategy_Development": "Long-term goals, Vision80, Win26, competitive landscape, achievements, awards, ESG, geopolitical impact, privacy",
    "Major_Transactions_Agreements": "Equity transfers, transactions, share repurchases, distribution agreements, convertible bonds, policy support, regulations"
}

def get_last_processed_id():
    """Read the last processed ID from file or return 0 if file doesn't exist"""
    try:
        with open(LAST_PROCESSED_ID_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def save_last_processed_id(last_id):
    """Save the last processed ID to file"""
    with open(LAST_PROCESSED_ID_FILE, 'w') as f:
        f.write(str(last_id))

def classify_question(question, is_rag):
    """
    Classify a question into one of the predefined categories using LLM
    
    Args:
        question: The question text to classify
        is_rag: Integer flag indicating if the question was processed with RAG (1) or not (0)
    """
    if is_rag == 0 or not question:
        return "non_rag"
    categories_description = "\n".join([f"{idx+1}. {cat}: {CATEGORY_DESCRIPTIONS[cat]}" 
                                       for idx, cat in enumerate(CATEGORIES)])
    
    prompt = f"""Classify the following question into exactly one of these categories:

{categories_description}

Question: {question}

Return ONLY the category name without any explanation or additional text. For example, if the question is about headquarters, just return "Company_Basics_Governance". Do not include numbers, punctuation, or anything else."""
    
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies questions into predefined categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            category = response.choices[0].message.content.strip()
            
            if category in CATEGORIES:
                return category
            else:
                for valid_cat in CATEGORIES:
                    if valid_cat.lower() in category.lower():
                        return valid_cat
                logger.warning(f"Invalid category returned for question: '{question}'. LLM returned: '{category}'. Defaulting to non_rag.")
                return "non_rag"
                
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Error during API call: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error(f"Failed to classify question after {retries} attempts: {question}")
                logger.error(f"Error: {e}")
                return "non_rag"


def calculate_jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts and determine if they match"""
    words1 = set(re.findall(r'\b\w+\b', text1))
    words2 = set(re.findall(r'\b\w+\b', text2))
    
    if not words1 or not words2:
        return False, 0.0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union > 0:
        similarity = intersection / union
        is_match = similarity >= 0.6
        return is_match, similarity
    else:
        return False, 0.0

def is_semantic_match(question1, question2):
    if not question1 or not question2:
        return False, 0.0
    norm_q1 = question1.lower().strip()
    norm_q2 = question2.lower().strip()
    
    if norm_q1 == norm_q2:
        return True, 1.0
    
    # Semantic matching
    prompt = f"""Determine if these two questions are asking for the same information, even if phrased differently:

Question 1: {question1}
Question 2: {question2}

First, analyze both questions to understand what information each is seeking.
Then determine if they are asking for the same information or different information.
Reply with ONLY one word: either "yes" (they are semantically equivalent) or "no" (they are different questions).
"""
    
    retries = 2
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if questions are asking for the same information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "yes" in result:
                confidence = 0.9
                return True, confidence
            elif "no" in result:
                confidence = 0.1
                return False, confidence
            else:
                if attempt < retries - 1:
                    continue
                else:
                    # Fall back to Jaccard similarity
                    return calculate_jaccard_similarity(norm_q1, norm_q2)
                
        except Exception as e:
            logger.warning(f"Error during semantic matching API call: {e}. Falling back to text similarity.")
            return calculate_jaccard_similarity(norm_q1, norm_q2)
            
    # Fall back to Jaccard similarity
    return calculate_jaccard_similarity(norm_q1, norm_q2)
    

    

def find_matching_qa_id(question, freq_qa_conn):
    if not question:
        return None, 0.0
        
    cursor = freq_qa_conn.cursor()
    
    cursor.execute("SELECT id, question FROM frequent_qa_pairs WHERE is_active = TRUE")
    qa_pairs = cursor.fetchall() # Get all questions
    
    if not qa_pairs:
        return None, 0.0
    
    best_match_id = None
    best_match_score = 0.0
    
    # Jaccard similarity as pre-filter
    candidates = []
    norm_question = question.lower().strip()
    
    for qa_id, qa_question in qa_pairs:
        norm_qa_question = qa_question.lower().strip()
        if norm_question == norm_qa_question:
            return qa_id, 1.0
            
        is_match, score = calculate_jaccard_similarity(norm_question, norm_qa_question)
        
        if score > 0.3: 
            candidates.append((qa_id, qa_question, score))
    
    # Sort and take top 5
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:5]
    
    logger.debug(f"Found {len(top_candidates)} preliminary candidates for question: {question[:50]}...")
    
    # Semantic matching using LLM
    for qa_id, qa_question, preliminary_score in top_candidates:
        is_match, confidence = is_semantic_match(question, qa_question)
        
        if is_match and confidence > best_match_score:
            best_match_id = qa_id
            best_match_score = confidence
            
            # Strong match -> no need to check further
            if confidence > 0.9:
                break
    
    # Return the best match if confidence exceeds threshold
    if best_match_score >= 0.7: 
        return best_match_id, best_match_score
    else:
        return None, best_match_score

def process_feedback_records():
    """Process new records from feedback table and insert into feedback_question_aliases"""
    last_id = get_last_processed_id()
    current_max_id = last_id
    
    try:
        feedback_conn = sqlite3.connect(FEEDBACK_DB_PATH)
        feedback_cursor = feedback_conn.cursor()
        
        frequent_qa_conn = sqlite3.connect(FREQUENT_QA_DB_PATH)
        frequent_qa_cursor = frequent_qa_conn.cursor()
        
        feedback_cursor.execute(
            "SELECT id, session_id, response_id, rating, question, response, is_rag, created_at FROM feedback WHERE id > ? ORDER BY id",
            (last_id,)
        )
        
        records = feedback_cursor.fetchall()
        
        if not records:
            logger.info("No new records to process")
            return
            
        logger.info(f"Found {len(records)} new records to process")
        
        # Process each record; save the last processed record ID
        for record in records:
            id, session_id, response_id, rating, question, response, is_rag, created_at = record
            if id > current_max_id:
                current_max_id = id
                
            try:
                category = classify_question(question, is_rag)
                
                # Check if question matches any existing QA pair 
                qa_id = None
                match_confidence = 0.0
                if is_rag == 1:
                    qa_id, match_confidence = find_matching_qa_id(question, frequent_qa_conn)
                
                is_match = qa_id is not None 
                # Using the original question as alias text if it's a match
                alias_text = question if is_match else None
                
                frequent_qa_cursor.execute('''
                INSERT INTO feedback_question_aliases 
                (qa_id, alias_text, session_id, response_id, rating, question, answer, category, 
                 is_match, match_confidence, created_at, notes) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    qa_id, 
                    alias_text,
                    session_id,
                    response_id,
                    rating,
                    question,
                    response,
                    category,
                    is_match,
                    match_confidence,
                    created_at,
                    f"Processed from feedback database. is_rag={is_rag}"
                ))
                
                logger.debug(f"Processed record ID {id}: Question classified as '{category}', Match confidence: {match_confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing record ID {id}: {e}")
                continue
            
        frequent_qa_conn.commit()
        
        # Update the last processed ID
        save_last_processed_id(current_max_id)
        logger.info(f"Successfully processed {len(records)} records. Last processed ID: {current_max_id}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        if 'feedback_conn' in locals():
            feedback_conn.close()
        if 'frequent_qa_conn' in locals():
            frequent_qa_conn.close()

def main():
    """Main function to run the feedback processor periodically"""
    logger.info("Starting Feedback Processor")
    
    while True:
        try:
            start_time = time.time()
            logger.info("Beginning processing cycle")
            
            process_feedback_records()
            
            elapsed = time.time() - start_time
            wait_time = max(1, PROCESSING_INTERVAL_SECONDS - elapsed)
            
            logger.info(f"Processing complete. Waiting {wait_time:.2f} seconds until next cycle.")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.info("Waiting 60 seconds before retrying...")
            time.sleep(60)

if __name__ == "__main__":
    main()