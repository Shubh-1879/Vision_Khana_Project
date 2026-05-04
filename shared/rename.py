import os
data_path = 'dataset/khana/'
for root, dirs, files in os.walk(data_path):
    for file in files:
        if '.' not in file:  # no extension
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, file + '.jpg')
            os.rename(old_path, new_path)
print('Renamed files to .jpg')