import os

folder_path = './UAV-benchmark-M'
file_names = os.listdir(folder_path)
file_dict = {}

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    #print("file_path is ", file_path)
    count = 0
    for file in os.listdir(file_path):
        count += 1
    file_dict[file_name] = count
    print(file_name, " : ", count)

print("sum is ", sum(file_dict.values()))