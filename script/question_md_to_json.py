import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

def process_md_file(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    for line in content.split('\n'):
        if line.strip():  
            questions.append({"question": line.strip()})
    return questions


def save_to_json(questions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

def process_all_files(directory_path):
    processed_count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.md') and not "README" in filename:
            md_file_path = os.path.join(directory_path, filename)
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_file_path = os.path.join(directory_path, json_filename)
            questions = process_md_file(md_file_path)
            save_to_json(questions, json_file_path)
            processed_count += 1
            print(f"Processed {md_file_path} -> {json_file_path} ({len(questions)} questions)")
    return processed_count

if __name__ == "__main__":
    question_directory = os.path.join(script_dir, '../src/test/test_questions')
    file_count = process_all_files(question_directory)
    print(f"Successfully processed {file_count} .md files")