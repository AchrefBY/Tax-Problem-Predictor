import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


pd.set_option('display.max_columns', None)

# Load the dataset
data = pd.read_csv("WorkingDB.csv")

# Encode categorical features
label_encoder = LabelEncoder()
data['type_de_donnée_encoded'] = label_encoder.fit_transform(data['type_de_donnée'])
data.drop(['type_de_donnée'], axis=1, inplace=True)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['industrie','catégorie_d\'actifs'])

print(data.head())

# Split the dataset into training and testing sets
train_data = data[data['type_de_donnée_encoded'] == 1]
test_data = data[data['type_de_donnée_encoded'] == 0]

X_train = train_data.drop(['difficulté_fiscale', 'type_de_donnée_encoded'], axis=1)
y_train = train_data['difficulté_fiscale']

X_test = test_data.drop(['difficulté_fiscale', 'type_de_donnée_encoded'], axis=1)
y_test = test_data['difficulté_fiscale']

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Get the feature importances 
feature_importances = rf_classifier.feature_importances_
print(feature_importances)

# Predict the tax_problem values on the testing data
predictions = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Trained model (replace with your actual trained model)
trained_rf_model = rf_classifier

# save the order of the columns for later use (when we will need to reorder the columns of the user input to match the model's expected input) 
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# Save the trained model to a .pkl file
joblib.dump(trained_rf_model, 'trained_rf_model.pkl')
