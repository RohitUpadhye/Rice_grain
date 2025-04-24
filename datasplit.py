import os
import shutil
import random


SOURCE_DIR = r"C:\Users\HP\Downloads\archive (15)\Rice_Image_Dataset"
 
DEST_DIR = "data"  #  split folders will be created

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset():
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIOS['train'] * total)
        val_end = train_end + int(SPLIT_RATIOS['val'] * total)

        split_data = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split in ["train", "val", "test"]:
            split_dir = os.path.join(DEST_DIR, split, class_name)
            create_dir(split_dir)
            for img_name in split_data[split]:
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(split_dir, img_name)
                shutil.copyfile(src_path, dst_path)

    print("✅ Dataset successfully split into train, val, and test folders!")

if __name__ == "__main__":
    split_dataset()
