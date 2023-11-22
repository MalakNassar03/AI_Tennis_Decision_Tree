#Malak Nassar
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create a DataFrame with the provided data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Specify the target variable column name
target_column_name = "Play Tennis"

# Split the dataset into features (X) and the target variable (y)
X = data.drop(target_column_name, axis=1)
y = data[target_column_name]

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier
clf.fit(X, y)

# Convert the class names to a list
class_names = label_encoders[target_column_name].classes_.tolist()

# Convert the DataFrame index to a list of feature names
feature_names = X.columns.tolist()

# Visualize the decision tree
plt.figure(figsize=(15, 8))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()