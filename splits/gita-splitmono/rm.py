#!/usr/bin/env python3
# filepath: /home/yzhong/gits/interpretable-pd/splits/gita-splitmono/rm.py

import os
import pandas as pd
import glob

def process_csv_files(root_dir):
    """
    Process all CSV files in the specified directory and its subdirectories,
    keeping only the header row and rows where the third column is 'MONOLOGUE'
    """
    # Find all CSV files in subdirectories
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            print(f"Processing file: {file_path}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Ensure the file has at least 3 columns
            if df.shape[1] < 3:
                print(f"Warning: {file_path} has fewer than 3 columns, skipping")
                continue
            
            # Get the header row
            header = df.columns
            
            # Filter rows where the third column is 'MONOLOGUE'
            monologue_rows = df[df.iloc[:, 2] == 'MONOLOGUE']
            
            # Create a new DataFrame with only the header and MONOLOGUE rows
            result_df = pd.DataFrame(columns=header)
            result_df = pd.concat([result_df, monologue_rows])
            
            # Write the results back to the original file
            result_df.to_csv(file_path, index=False)
            
            print(f"File processing complete: {file_path}")
            print(f"Kept {len(result_df)} MONOLOGUE rows + 1 header row")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

if __name__ == "__main__":
    # Specify directory path
    base_dir = './'
    print(f"Starting to process directory: {base_dir}")
    
    # Process CSV files
    process_csv_files(base_dir)
    
    print("All CSV files processing complete")