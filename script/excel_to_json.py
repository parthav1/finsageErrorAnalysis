import csv
import json

# 把准备好的表格转化成json 以便存入向量数据库
def convert_csv_to_json(input_path, output_path):
    result = []

    data_columns = [
        'Y2023_Q1', 'Y2023_Q2', 'Y2023_H1', 'Y2023_Q3', 'Y2023_Q4', 
        'Y2023_H2', 'Y2023_FY', 'Y2024_Q1', 'Y2024_Q2', 'Y2024_H1', 
        'Y2024_Q3', 'Y2024_Q4', 'Y2024_H2', 'Y2024_FY'
    ]
    
    try:
        with open(input_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Create the data dictionary with only non-empty values
                data_dict = {}
                for col in data_columns:
                    value = row.get(col, '').strip()
                    # Keep the value as is, including empty strings
                    data_dict[col] = value
                
                # json structure
                json_row = {
                    "question": row.get('question', '').strip(),
                    "question_rewritten": row.get('question_rewritten', '').strip(),
                    "data": data_dict
                }
                
                result.append(json_row)
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(result, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted {len(result)} rows from {input_path} to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    input_csv_path = "/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/updated.csv"  
    output_json_path = "/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/input.json"  
    convert_csv_to_json(input_csv_path, output_json_path)
