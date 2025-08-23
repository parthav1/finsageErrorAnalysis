import os
import re

def convert_dir_to_md(input_dir, output_file):
    # Open the output markdown file in write mode
    cnt = 0
    with open(output_file, 'w', encoding='utf-8') as md_file:
        # Walk through each directory and subdirectory
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                # Process only text files (or modify as needed)
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Split content based on '******* Question X *******' format
                    questions = content.split("******* Question ")
                    
                    # Skip the first split part if it's empty
                    questions = questions[1:]
                    
                    # Write structured output in Markdown format for each file
                    # md_file.write(f"# File: {filename} (Path: {file_path})\n\n")  # Add filename and path as header
                    for question_block in questions:
                        # Extract question number
                        block = question_block.split('---Question---')[1]
                        
                        

                        question = block.split('---Rewritten Question---')[0].strip()
                        block = block.split('---Rewritten Question---')[1]
                        # rewrite_question = block.split('---Need RAG---')[0].strip()
                        # block = block.split('---Need RAG---')[1]
                        rewrite_question = block.split('---History Summary---')[0].strip()
                        block = block.split('---History Summary---')[1]
                        history = block.split('---RAG Info (DataFrame)---')[0].strip()
                        block = block.split('---RAG Info (DataFrame)---')[1]
                        rag_info = block.split('---Answer---')[0].strip()
                        answer = block.split('---Answer---')[1].replace('\n', '')

                        cnt += 1
                        md_file.write(f"## Question {cnt}\n\n")
                        md_file.write(f"**Question:** {question.strip()}\n\n")
                        md_file.write(f"**Rewritten Question:** {rewrite_question.strip()}\n\n")
                        # md_file.write(f"**Need RAG:** {need_rag.strip()}\n\n")
                        # md_file.write(f"**History Summary:** {history.strip()}\n\n")
                        # md_file.write(f"**RAG Info (DataFrame):** {rag_info.strip()}\n\n")
                        md_file.write(f"**Answer:** {answer.strip()}\n\n")
                        md_file.write("---\n\n")

    print(f"All files in '{input_dir}' have been merged into '{output_file}'")

# Specify the input directory and output file
input_dir = '/root/autodl-tmp/RAG_Agent_vllm/src/test/test_questions/question_all40_output'  # Replace with the path to your directory
output_file = 'merged_question_output.md'  # Specify the output markdown file name

# Convert all files in directory to Markdown
convert_dir_to_md(input_dir, output_file)