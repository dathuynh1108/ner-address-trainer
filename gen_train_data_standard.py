import json
import re
import random
import csv


DISTRICT_PREFIX_REGEX = re.compile(
    r"^(?:q\.?\s?\d*|quan|quận|h\.?\s?|huyen|huyện|tp\.?|t\.p\.?|thanh pho|thành phố|thi xa|thị xã|tx\.?\s?)\b\.?,?\s*",
    flags=re.IGNORECASE,
)

WARD_PREFIX_REGEX = re.compile(
    r"^(?:p\.?\s?\d*|phuong|phường|xa|xã|x\.?|đặc khu|dac khu|dk\.?|thi tran|thị trấn|tt\.?|khu pho|khu phố|kp\.?)\b\.?,?\s*",
    flags=re.IGNORECASE,
)

PROVINCE_PREFIX_REGEX = re.compile(
    r"^(?:tp\.?\s?|t\.?\s?|thanh pho|thành phố|tinh|tỉnh)\b\.?,?\s*",
    flags=re.IGNORECASE,
)

def generate_address_variations_with_separator(province, district, ward, separator=","):
    """Generate address format variations with custom separator"""  
    def remove_prefix(text, regex):
        if not text:
            return text
        
        # Check if the text contains numbers - if so, don't remove prefix
        if re.search(r'\d', text):
            return text
        
        return regex.sub('', text).strip()
    
    # Create trimmed versions
    province_trimmed = remove_prefix(province, PROVINCE_PREFIX_REGEX)
    district_trimmed = remove_prefix(district, DISTRICT_PREFIX_REGEX)
    ward_trimmed = remove_prefix(ward, WARD_PREFIX_REGEX)
    
    # Generate variations with original names
    original_variations = [
        f"{ward}{separator} {district}{separator} {province}",
        f"{province}{separator} {district}{separator} {ward}",
        f"{district}{separator} {ward}{separator} {province}",
        f"{district}{separator} {province}"
    ]
    
    # Generate variations with trimmed names
    trimmed_variations = [
        f"{ward_trimmed}{separator} {district_trimmed}{separator} {province_trimmed}",
        f"{province_trimmed}{separator} {district_trimmed}{separator} {ward_trimmed}",
        f"{district_trimmed}{separator} {ward_trimmed}{separator} {province_trimmed}",
        f"{district_trimmed}{separator} {province_trimmed}"
    ]
    
    # Generate mixed variations (some with prefix, some without)
    mixed_variations = [
        f"{ward_trimmed}{separator} {district}{separator} {province_trimmed}",
        f"{ward}{separator} {district_trimmed}{separator} {province}",
        f"{ward_trimmed}{separator} {district_trimmed}{separator} {province}",
        f"{ward}{separator} {district}{separator} {province_trimmed}"
    ]
    
    # Return as tuples with corresponding labels
    result = []
    
    # Add original variations with original labels
    for var in original_variations:
        if var.strip() and not var.startswith(separator) and not var.endswith(separator):
            result.append((var, province, district, ward))
    
    # Add trimmed variations with trimmed labels
    for var in trimmed_variations:
        if var.strip() and not var.startswith(separator) and not var.endswith(separator):
            result.append((var, province_trimmed, district_trimmed, ward_trimmed))
    
    # Add mixed variations with appropriate labels
    for var in mixed_variations:
        if var.strip() and not var.startswith(separator) and not var.endswith(separator):
            # Use mixed labels based on which version was used in the address
            if ward_trimmed in var and district_trimmed in var:
                result.append((var, province, district_trimmed, ward_trimmed))
            elif ward in var and district in var:
                result.append((var, province_trimmed, district, ward))
            else:
                result.append((var, province, district, ward))
    
    return result

def parse_administrative_csv(csv_file_path):
    """Parse CSV file containing administrative data"""
    administrative_units = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
            # Try to detect if file has headers
            sample = file.read(1024)
            file.seek(0)
            
            # Check if first line looks like headers
            first_line = file.readline().strip()
            file.seek(0)
            
            has_header = any(keyword in first_line.lower() for keyword in ['province', 'district', 'ward', 'tỉnh', 'huyện', 'phường', 'xã'])
            
            csv_reader = csv.reader(file)
            
            if has_header:
                next(csv_reader)  # Skip header row
            
            for row in csv_reader:
                if len(row) >= 3:
                    province = row[0].strip()
                    district = row[1].strip()
                    ward = row[2].strip()
                    
                    if province and district and ward:
                        administrative_units.append((province, district, ward))
                        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found!")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    return administrative_units

# Parse administrative data from CSV file
csv_file_path = 'data/standard_addresses.csv'
administrative_units = parse_administrative_csv(csv_file_path)

if not administrative_units:
    print("No administrative units found. Please check your CSV file.")
    exit(1)

print(f"Loaded {len(administrative_units)} administrative units from CSV")

# Generate training data from administrative units
generated_data = []
for province, district, ward in administrative_units:
    separators = [",", ";", ""]
    
    variations = []
    vistied_addresses = set()
    for sep in separators:
        variations = generate_address_variations_with_separator(province, district, ward, sep)
        for address, prov_label, dist_label, ward_label in variations:
            if address not in vistied_addresses:
                vistied_addresses.add(address)
                
                generated_data.append({
                    'address': address,
                    'province': prov_label,
                    'district': dist_label,
                    'ward': ward_label
                })

# Read existing JSON data if it exists
try:
    with open('data/addresses.json', 'r', encoding='utf-8-sig') as f:
        existing_data = json.load(f)
except FileNotFoundError:
    existing_data = []

# Combine existing data with generated data
all_data = existing_data + generated_data

# Shuffle the combined data
random.shuffle(all_data)

print("Generated", len(generated_data), "training examples")

# Write combined data back to JSON
with open('data/addresses.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)