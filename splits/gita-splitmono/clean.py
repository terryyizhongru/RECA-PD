import os
import pandas as pd

# List of sample_ids to remove
sample_ids_to_remove = [
    "HC_MONOLOGUE_AVPEPUDEAC0039_-MONOLOGO-NR_part7",
    "HC_MONOLOGUE_AVPEPUDEAC0040_-MONOLOGO-NR_part8",
    "PD_MONOLOGUE_AVPEPUDEA0035_-MONOLOGO-NR_part3"
]

# Base directory path
base_directory = "./"

# List to store all found CSV files
all_csv_files = []

# Recursively search for CSV files in all subdirectories
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.csv'):
            all_csv_files.append(os.path.join(root, file))

print(f"Found {len(all_csv_files)} CSV files")

# Process each CSV file
for file_path in all_csv_files:
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Check if there's a sample_id column
        if 'sample_id' in df.columns:
            # Filter out specific sample_id rows
            original_row_count = len(df)
            df = df[~df['sample_id'].isin(sample_ids_to_remove)]
            removed_row_count = original_row_count - len(df)
            
            if removed_row_count > 0:
                print(f"{file_path}: Removed {removed_row_count} rows")
        
        # Replace path strings in all columns
        for column in df.columns:
            if df[column].dtype == 'object':  # Only process string columns
                df[column] = df[column].astype(str).str.replace('./data/gita/', './data/gita-splitmono/')
        
        # Save the modified CSV file
        df.to_csv(file_path, index=False)
        print(f"Processed: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("All CSV files processing complete!")


# Base directory path
base_directory = "./"

# List to store all found CSV files
all_csv_files = []

# Recursively search for CSV files in all subdirectories
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.csv'):
            all_csv_files.append(os.path.join(root, file))

print(f"Found {len(all_csv_files)} CSV files")

# Process each CSV file
total_replacements = 0

for file_path in all_csv_files:
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        file_replacements = 0
        
        # Process each column
        for column in df.columns:
            if df[column].dtype == 'object':  # Only process string columns
                # For each part number (1-10), replace _partX_ with __partX
                for i in range(1, 11):
                    old_pattern = f"_part{i}_"
                    new_pattern = f"__part{i}"
                    
                    # Count replacements in each column
                    col_replacements = df[column].astype(str).str.count(old_pattern).sum()
                    file_replacements += col_replacements
                    
                    # Perform replacement
                    df[column] = df[column].astype(str).str.replace(old_pattern, new_pattern, regex=False)
        
        total_replacements += file_replacements
        
        # Save the modified CSV file
        df.to_csv(file_path, index=False)
        print(f"Processed: {file_path} (Replaced {file_replacements} instances)")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"All CSV files processing complete! Total of {total_replacements} '_partX_' replaced with '__partX'")