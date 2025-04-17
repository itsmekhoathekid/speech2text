import json
import os
import csv

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None

def save_csv(data, save_path):
    """
    Lưu dữ liệu thành file CSV.
    """
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(save_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["#id", "char"])  # Tiêu đề cột
        for char, id in sorted(data.items(), key=lambda x: x[1]):
            writer.writerow([id, char])
    
    print(f"File CSV đã được lưu tại: {save_path}")

def process_data(json_path, save_csv_path):
    data = load_json(json_path)
    if not data:
        return
    
    save_csv(data, save_csv_path)

# Ví dụ sử dụng:
process_data("/data/npl/Speech2Text/Speech-Transformer/data/dic.json", "/data/npl/Speech2Text/RNN-Transducer/data/dic.labels")