import json
import time
from typing import List, Dict, Any
from openai import OpenAI

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

API_KEY = "sk-BUYcNLayB5w9rHwa1gYodJq3bgbJX6ChWNeRBNe1I6CgX4zr"
client = OpenAI(api_key=API_KEY,base_url="https://api.lkeap.cloud.tencent.com/v1")

def classify_question(question: str, model: str = "deepseek-v3"):
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
                model=model,
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

                print(f"Warning: Invalid category returned for question: '{question}'. LLM returned: '{category}'. Defaulting to {CATEGORIES[0]}.")
                return CATEGORIES[0]
                
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error during API call: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Failed to classify question after {retries} attempts: {question}")
                print(f"Error: {e}")
                return "Unclassified"  # Default fallback

def classify_questions_from_file(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(questions)} questions from {input_file}")
    results = {category: [] for category in CATEGORIES}
    results["Unclassified"] = []
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}{'...' if len(question) > 50 else ''}")
        
        category = classify_question(question)
        
        if category in results:
            results[category].append(question)
        else:
            results["Unclassified"].append(question)
            
        # Add delay to avoid rate limiting
        if (i + 1) % 5 == 0:
            time.sleep(1)
    
    output_data = {
        "total_questions": len(questions),
        "categories": {
            category: {
                "count": len(questions_list),
                "questions": questions_list
            }
            for category, questions_list in results.items() if questions_list
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Classification completed. Results written to {output_file}")
    print("\nClassification Summary:")
    for category, questions_list in results.items():
        if questions_list:
            print(f"{category}: {len(questions_list)} questions")

INPUT_FILE = "/root/autodl-tmp/dir_tzh/dev/RAG_Agent/src/test/test_questions/all_questions.md"
OUTPUT_FILE = "classified_questions.json"

def main():
    classify_questions_from_file(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()