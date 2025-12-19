import json
import os
import re
from tqdm import tqdm

input_dir =  "./training_samples/text/"
output_dir = "./training_samples/prompt/"

os.makedirs(output_dir, exist_ok=True)

# get file list
files = os.listdir(input_dir)

for i in tqdm(files, desc="Processing files"):
    file_path = os.path.join(input_dir, i)

    # read and clean
    with open(file_path, 'r') as file:
        data = json.load(file)
        cleaned_dict = {}
        stop_processing = False
        
        for k, v in data.items():
            if stop_processing:
                break
            
            if v and str(v).strip():  # Check if value is valid
                new_key = re.sub(r'^\d+\.\s*', '', k)  # Clean key
                new_value = re.sub(r'^\d+\.\s*', '', v)  # Clean value
                
                # Replace specific values
                if new_value == 'overexposed' or new_value == 'underexposed' or new_value == 'overexposure' or new_value == 'underexposure':
                    new_value = 'proper exposure'
                
                # Check for 'other' in key before adding it
                if 'other' in new_key:
                    stop_processing = True
                else:
                    cleaned_dict[new_key] = new_value

    # write to a new directory
    output_file_path = os.path.join(output_dir, i)
    print(cleaned_dict)
    with open(output_file_path, 'w') as file:
        json.dump(cleaned_dict, file, indent=4)


