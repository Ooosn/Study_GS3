import os
from collections import defaultdict

# 你要统计的扩展名
exts = [".py", ".cu",  ".h", ".cpp", "sh"]

# 初始化：每种扩展名的行数总计
line_counts = defaultdict(int)
file_counts = defaultdict(int)

for root, _, files in os.walk("."):
    for file in files:
        for ext in exts:
            if file.endswith(ext):
                try:
                    full_path = os.path.join(root, file)
                    # 如果路径中包含 third_party，则跳过该文件
                    if "third_party" in full_path:
                        continue
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        line_counts[ext] += line_count
                        file_counts[ext] += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")

# 输出结果
print("📊 Line Count by File Type:\n")
total = 0
for ext in exts:
    print(f"{ext:5} | {file_counts[ext]:4d} files | {line_counts[ext]:7d} lines")
    total += line_counts[ext]

print(f"\n🧮 Total lines across all: {total}")
