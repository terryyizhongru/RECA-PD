#!/usr/bin/env python3
# filepath: /home/yzhong/gits/interpretable-pd/exps/gita/cross_full_newcate/calculate_mean.py

import re

def calculate_average_from_file(file_path):
    f1_values = []
    std_values = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Skip comment lines or empty lines
                if line.strip().startswith('//') or not line.strip():
                    continue
                
                # Extract F1 and standard deviation values using regex
                match = re.search(r'F1:\s+(\d+\.\d+)\s+±\s+(\d+\.\d+)', line)
                if match:
                    f1 = float(match.group(1))
                    std = float(match.group(2))
                    f1_values.append(f1)
                    std_values.append(std)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
    # Calculate means
    if f1_values:
        f1_mean = sum(f1_values) / len(f1_values)
        std_mean = sum(std_values) / len(std_values)
        return f1_mean, std_mean
    else:
        return None, None

def main(file_path):
    f1_mean, std_mean = calculate_average_from_file(file_path)
    
    if f1_mean is not None and std_mean is not None:
        print(f"Average F1 +- std: {f1_mean:.4f} ± {std_mean:.4f}")
    else:
        print("No valid data found in the file.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])