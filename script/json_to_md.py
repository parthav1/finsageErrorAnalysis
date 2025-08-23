import json
from pathlib import Path
import os
import pandas as pd
import markdown


def json_to_markdown(file_path, file_name):
    file = Path(file_path)
    with open(f"{file}", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    #print(df['question'].values)

    markdown_table = '\n'.join(list(df['question'].values))
    #print(markdown_table)

    with open(f"{file.parent}/{file_name}.md", "w") as f:
        f.write(markdown_table)


json_to_markdown('/root/autodl-tmp/RAG_Agent_production/src/test/test_questions/zeekr_questions/zeekr_20.json', 'question_batch1')

