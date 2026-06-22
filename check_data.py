import csv

counts = {}

with open('hand_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0:
            label = row[-1]
            counts[label] = counts.get(label, 0) + 1

for letter, count in sorted(counts.items()):
    print(f"Letter {letter}: {count} samples")

print(f"Total: {sum(counts.values())} samples")