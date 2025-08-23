import json
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, List

def collect_metrics(root_dir: str) -> Dict[str, Dict]:
    """
    Collect metrics from metrics.json files in all subdirectories.
    
    Args:
        root_dir (str): Root directory to start searching from
        
    Returns:
        Dict[str, Dict]: Dictionary mapping directory names to their metrics
    """
    metrics_by_dir = {}
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'result_2.json' in filenames:
            try:
                # Read metrics.json file
                metrics_path = Path(dirpath) / 'result_2.json'
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                
                # Get first dictionary if metrics_data is a list
                if isinstance(metrics_data, list):
                    metrics_data = metrics_data[0]
                    # remove "avg_ppl" key from metrics_data
                    if "avg_ppl" in metrics_data:
                        del metrics_data["avg_ppl"]
                
                # Use directory name as key
                dir_name = os.path.basename(dirpath)
                metrics_by_dir[dir_name] = metrics_data
                
            except Exception as e:
                print(f"Error reading metrics from {dirpath}: {e}")
    
    return metrics_by_dir

def create_metrics_table(metrics_by_dir: Dict[str, Dict]) -> None:
    """
    Create and save an HTML table visualization of the metrics.
    
    Args:
        metrics_by_dir (Dict[str, Dict]): Dictionary mapping directory names to their metrics
    """
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(metrics_by_dir, orient='index')
    
    # Round numeric values to 4 decimal places
    df = df.round(4)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Retriever'] + list(df.columns),
            fill_color='white',
            align=['left'] + ['center'] * len(df.columns),  # Left align text, center align numbers
            font=dict(
                size=14,
                color='black',
                family='Times New Roman',
            ),
            line=dict(color='black', width=1.5)
        ),
        cells=dict(
            values=[df.index] + [df[col] for col in df.columns],
            fill_color=[['white', '#f5f5f5'] * (len(df)//2 + 1)],  # Alternating row colors
            align=['left'] + ['center'] * len(df.columns),  # Left align text, center align numbers
            font=dict(
                size=12,
                color='black',
                family='Times New Roman'
            ),
            line=dict(color='black', width=1),
            height=30
        )
    )])
    
    # Update layout for academic paper style
    fig.update_layout(
        title=dict(
            text='Metrics Comparison Across Directories',
            font=dict(
                size=16,
                family='Times New Roman',
                color='black'
            ),
            x=0.5
        ),
        width=1200,
        height=max(400, len(metrics_by_dir) * 30 + 100),
        margin=dict(t=50, l=50, r=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Save as HTML file
    fig.write_html('metrics_comparison.html')
    print("Table has been saved as 'metrics_comparison.html'")
    
    # Save as PDF file
    fig.write_image("metrics_comparison.pdf")
    print("Table has been saved as 'metrics_comparison.pdf'")

def main():
    # Get root directory from user input
    root_dir = "/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_retrievers/lora_75_75"
    
    # Collect metrics
    print("Collecting metrics...")
    metrics_by_dir = collect_metrics(root_dir)
    
    if not metrics_by_dir:
        print("No metrics.json files found!")
        return
    
    # Create and save table
    print(f"Found metrics from {len(metrics_by_dir)} directories. Creating table...")
    create_metrics_table(metrics_by_dir)

if __name__ == "__main__":
    main()