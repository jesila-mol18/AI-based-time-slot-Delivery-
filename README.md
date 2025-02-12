import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
df = pd.read_csv("/content/drive/MyDrive/delivery_data.csv")

# Convert categorical values to numerical labels
mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3, "Low": 0, "Medium": 1, "High": 2}
df.replace(mapping, inplace=True)

# Features and target
X = df[["day_of_week", "previous_successful_slot", "delivery_urgency", "work_from_home", "delivery_distance_km", "delivery_frequency"]]
y = df["preferred_time_slot"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the AI model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict delivery slot
def predict_delivery_slot(day_of_week, previous_slot, urgency, wfh, distance, frequency):
    input_data = np.array([[day_of_week, previous_slot, urgency, wfh, distance, frequency]])
    prediction = model.predict(input_data)
    slot_mapping = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}
    return slot_mapping[prediction[0]]

# Example usage
example_slot = predict_delivery_slot(day_of_week=2, previous_slot=1, urgency=2, wfh=1, distance=10, frequency=3)
print(f"Predicted Best Delivery Slot: {example_slot}")

from google.colab import drive
drive.mount('/content/drive')

output:
Model Accuracy: 0.20
Predicted Best Delivery Slot: Morning
