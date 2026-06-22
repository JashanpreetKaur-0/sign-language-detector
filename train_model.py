import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the data from our CSV file
data = []
labels = []

with open('hand_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 64:           # 63 numbers + 1 label = 64 columns
            data.append(row[:63])    # first 63 = the hand coordinates
            labels.append(row[63])   # last one = the letter (A, B, C, D)

# Step 2: Convert to numbers (CSV stores everything as text)
data = np.array(data, dtype=float)
labels = np.array(labels)

print(f"Loaded {len(data)} samples")
print(f"Letters found: {set(labels)}")

# Step 3: Split into training data and testing data
# 80% used to teach the model, 20% used to test how well it learned
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

print(f"Training on {len(x_train)} samples")
print(f"Testing on {len(x_test)} samples")

# Step 4: Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100)
# n_estimators=100 means it builds 100 decision trees and votes
model.fit(x_train, y_train)   # this is where actual learning happens

# Step 5: Test how accurate it is
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Step 6: Show detailed results per letter
print("\nDetailed results:")
print(classification_report(y_test, predictions))

# Step 7: Save the trained model to a file
# so we can load it later in our real-time detector
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")