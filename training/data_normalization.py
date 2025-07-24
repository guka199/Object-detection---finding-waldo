import os

# Input and output directories
input_dir = 'training\data\labels\labels_without_normalization_for_waldo'
output_dir = 'training\data\labels\labels_normalized_for_waldo'
os.makedirs(output_dir, exist_ok=True)

# Fixed image dimensions
img_width = 64
img_height = 64

for label_file in os.listdir(input_dir):
    if not label_file.endswith('.txt'):
        continue

    input_path = os.path.join(input_dir, label_file)
    output_path = os.path.join(output_dir, label_file)

    normalized_lines = []

    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[ERROR] Skipping invalid line in {label_file}: {line}")
                continue

            cls, cx, cy, w, h = parts
            cx = float(cx) / img_width
            cy = float(cy) / img_height
            w = float(w) / img_width
            h = float(h) / img_height

            normalized_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(output_path, 'w') as f_out:
        f_out.write("\n".join(normalized_lines))

    print(f"[OK] Normalized: {label_file}")
