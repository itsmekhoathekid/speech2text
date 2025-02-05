# import json

# path_train1 = r"C:\Users\VIET HOANG - VTS\Desktop\data\train.json"
# path_train2 = r"C:\Users\VIET HOANG - VTS\Downloads\drive-download-20250127T045630Z-001\train.json"



# path_test1 = r"C:\Users\VIET HOANG - VTS\Desktop\data\test.json"
# path_test2 = r"C:\Users\VIET HOANG - VTS\Downloads\drive-download-20250127T045630Z-001\test.json"

# def join_data(path1, path2):
#     res = {}
#     with open(path1, 'r', encoding='utf-8') as f:
#         data1 = json.load(f)
#     with open(path2, 'r', encoding='utf-8') as f:
#         data2 = json.load(f)
#     for id, data in data1.items():
#         res[id] = data
#     for id, data in data2.items():
#         res[id] = data
#     return res

# def save_data(data, path):
#     with open(path, 'w') as f:
#         json.dump(data, f)

# def load_data(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         data2 = json.load(f)
#     return data2

# data_train = join_data(path_train1, path_train2)
# test_train = join_data(path_test1, path_test2)
# dev_train = load_data(r"C:\Users\VIET HOANG - VTS\Desktop\data\dev.json")

# save_data(data_train, r"C:\Users\VIET HOANG - VTS\Desktop\data\train.json")
# save_data(test_train, r"C:\Users\VIET HOANG - VTS\Desktop\data\test.json")
# save_data(dev_train, r"C:\Users\VIET HOANG - VTS\Desktop\data\dev.json")

# import pickle

# # Đường dẫn tới tệp .pkl
# pkl_file_path = r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\Speech-Transformer-master\data\data.pickle'

# # Đường dẫn tới tệp .txt
# txt_file_path = r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\Speech-Transformer-master\data\data.txt'

# # Đọc dữ liệu từ tệp .pkl
# with open(pkl_file_path, 'rb') as pkl_file:
#     data = pickle.load(pkl_file)

# # Ghi dữ liệu vào tệp .txt
# with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
#     txt_file.write(str(data))

import os
import shutil

# Định nghĩa đường dẫn thư mục nguồn và đích
source_dir = r"D:\voices\voices"
destination_dir = r"D:\wav-voices\wav-voices"

# Đảm bảo thư mục đích tồn tại
os.makedirs(destination_dir, exist_ok=True)

# Lặp qua tất cả các tệp trong thư mục nguồn
for file_name in os.listdir(source_dir):
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)

    # Kiểm tra nếu là file (tránh di chuyển thư mục con)
    if os.path.isfile(source_path):
        shutil.move(source_path, destination_path)
        print(f"Đã di chuyển: {file_name} → {destination_path}")

print("Di chuyển hoàn tất!")
