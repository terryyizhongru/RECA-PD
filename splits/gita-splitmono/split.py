#!/usr/bin/env python3

import os
import csv
import glob

def expand_csv_file(file_path):
    """
    Expand each non-header row in a CSV file into 10 rows,
    replacing 'MONOLOGO-NR' with 'MONOLOGO-NR_part1' through 'MONOLOGO-NR_part10'
    """
    # Read the original CSV file
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        print(f"Warning: Empty file {file_path}")
        return
    
    # Get the header row
    header = rows[0]
    
    # Process non-header rows
    new_rows = [header]  # Start with the header
    for row in rows[1:]:
        if not row:
            continue
        
        # Keep these columns unchanged
        subject_id = row[0]
        sample_id = row[1]
        task_id = row[2]
        label = row[3]
        
        # For each part (1-10), create a new row
        for part in range(1, 11):
            new_row = row.copy()
            
            # Replace MONOLOGO-NR with MONOLOGO-NR_part{part} in all columns
            for i in range(1, len(new_row)):
                if 'MONOLOGO-NR' in new_row[i]:
                    new_row[i] = new_row[i].replace('MONOLOGO-NR', f'MONOLOGO-NR_part{part}')
            
            new_rows.append(new_row)
    
    # Write the expanded rows back to the file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    
    print(f"Processed {file_path}: Expanded to {len(new_rows)} rows")

def process_directory(root_dir):
    """
    Process all CSV files in the given directory and its subdirectories
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                expand_csv_file(file_path)

if __name__ == "__main__":
    # Set the root directory
    root_dir = './'
    print(f"Processing CSV files in {root_dir}...")
    
    # Process all CSV files in the directory tree
    process_directory(root_dir)
    
    print("Processing complete!")