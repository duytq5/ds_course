import csv
from collections import defaultdict

# Analyze train.csv
print("="*50)
print("TRAIN.CSV ANALYSIS")
print("="*50)

with open('data/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    total = len(rows)
    
    missing = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if value == '' or value is None:
                missing[key] += 1
    
    print(f"\nTotal records: {total}")
    print("\nMissing values by column:")
    for col, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        pct = (count/total)*100
        print(f"  {col:15s}: {count:4d} ({pct:5.1f}%)")

# Analyze test.csv
print("\n" + "="*50)
print("TEST.CSV ANALYSIS")
print("="*50)

with open('data/test.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    total = len(rows)
    
    missing = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if value == '' or value is None:
                missing[key] += 1
    
    print(f"\nTotal records: {total}")
    print("\nMissing values by column:")
    for col, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        pct = (count/total)*100
        print(f"  {col:15s}: {count:4d} ({pct:5.1f}%)")

