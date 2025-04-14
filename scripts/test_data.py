import json
import glob
import os
from tqdm import tqdm

def extract_text_from_complicated_json(file_path, output_path=None):
    """
    Extract actual text content from JSON files with nested JSON in the text field.
    
    Args:
        file_path: Path to the JSON file
        output_path: Path to save the cleaned data (if None, append "_cleaned" to filename)
    """
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_cleaned{ext}"
    
    extracted_data = []
    valid_lines = 0
    invalid_lines = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the outer JSON object
                outer_json = json.loads(line)
                
                if "text" in outer_json:
                    text_content = outer_json["text"]
                    
                    # Check if text contains JSON inside backticks
                    if text_content.startswith("```json") and text_content.endswith("```"):
                        # Extract the JSON string inside the backticks
                        json_str = text_content.replace("```json", "", 1).replace("```", "", 1).strip()
                        
                        try:
                            # Parse the inner JSON
                            inner_json = json.loads(json_str)
                            
                            # Extract conversation content
                            if "conversations" in inner_json:
                                conversations = inner_json["conversations"]
                                
                                # Process each message in the conversation
                                actual_text = ""
                                for msg in conversations:
                                    if "from" in msg and "value" in msg:
                                        role = msg["from"]
                                        content = msg["value"]
                                        actual_text += f"{role}: {content}\n\n"
                                
                                # Create a clean JSON with the extracted conversation
                                clean_item = {"text": actual_text.strip()}
                                if "metadata" in outer_json:
                                    clean_item["metadata"] = outer_json["metadata"]
                                
                                extracted_data.append(clean_item)
                                valid_lines += 1
                            else:
                                # If no conversations field, use the inner JSON as is
                                clean_item = {"text": json_str}
                                if "metadata" in outer_json:
                                    clean_item["metadata"] = outer_json["metadata"]
                                extracted_data.append(clean_item)
                                valid_lines += 1
                        except json.JSONDecodeError:
                            # If inner JSON is invalid, use the text content as is
                            extracted_data.append({"text": text_content})
                            valid_lines += 1
                    else:
                        # Regular text field without nested JSON
                        extracted_data.append(outer_json)
                        valid_lines += 1
                else:
                    # No text field found
                    invalid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1
                print(f"Invalid JSON at line {line_num}")
    
    # Write the cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Processed file: {file_path}")
    print(f"Valid lines: {valid_lines}, Invalid lines: {invalid_lines}")
    print(f"Output saved to: {output_path}")
    
    return valid_lines, invalid_lines

def process_all_files(file_pattern):
    """Process all files matching the pattern"""
    files = glob.glob(file_pattern)
    total_valid = 0
    total_invalid = 0
    
    for file_path in tqdm(files, desc="Processing files"):
        valid, invalid = extract_text_from_complicated_json(file_path)
        total_valid += valid
        total_invalid += invalid
    
    print(f"\nTotal processing complete:")
    print(f"Total valid lines: {total_valid}")
    print(f"Total invalid lines: {total_invalid}")

# Usage
process_all_files("/scratch/vladimir_albrekht/projects/smollm/data/db_11_v2_splited_kazakh/deepseek_data/*.json")