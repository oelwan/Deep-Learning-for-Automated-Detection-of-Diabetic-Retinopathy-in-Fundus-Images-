import pandas as pd
import os
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/aptos2019/train.csv')

# Count occurrences of each diagnosis class
class_counts = df['diagnosis'].value_counts().sort_index()

# Print the class distribution
print('Class distribution:')
for class_label in range(5):  # Assuming 5 classes (0-4)
    count = class_counts.get(class_label, 0)
    percentage = (count / len(df)) * 100
    print(f'Class {class_label}: {count} images ({percentage:.2f}%)')

print(f'\nTotal images in CSV: {len(df)}')

# Check if all images exist
df['img_path'] = df['id_code'].apply(lambda x: f'data/aptos2019/train_images/{x}.png')
missing = [path for path in df['img_path'] if not os.path.exists(path)]

print(f'\nMissing images: {len(missing)}')
if missing:
    print('First few missing image IDs:')
    for path in missing[:5]:
        print(path)
else:
    print('All images exist!') 