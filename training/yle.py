import os

def fix_labels(folder_path):
    fixed_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Modify class index from '1' to '0'
            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '1':
                    parts[0] = '0'
                    changed = True
                new_lines.append(' '.join(parts) + '\n')

            # Only rewrite if changes were made
            if changed:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
    
    print(f"Fixed class indices in {fixed_count} files in {folder_path}")

# Replace these paths with your actual label directories
train_labels_path = r"C:\Users\zandro\Desktop\ML project\data\labels\train"
val_labels_path = r"C:\Users\zandro\Desktop\ML project\data\labels\val"

fix_labels(train_labels_path)
fix_labels(val_labels_path)

