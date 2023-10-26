import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
from sklearn.tree import export_graphviz
import graphviz
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import learning_curve


pd.set_option('display.max_columns', None)

# Load the dataset
data = pd.read_csv("DataBase.csv")

# Encode categorical features
label_encoder = LabelEncoder()
data['type_de_donnée_encoded'] = label_encoder.fit_transform(data['type_de_donnée'])
data.drop(['type_de_donnée'], axis=1, inplace=True)

data['industrie_encoded'] = label_encoder.fit_transform(data['industrie'])
data.drop(['industrie'], axis=1, inplace=True)

data['catégorie_d\'actifs_encoded'] = label_encoder.fit_transform(data['catégorie_d\'actifs'])
data.drop(['catégorie_d\'actifs'], axis=1, inplace=True)

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
feature_names = X_train.columns
feature_importance = rf_classifier.feature_importances_
print("Feature names:", feature_names)
print("Feature importances:", feature_importance)

# Visualize a decision tree from the Random Forest
tree_index_to_visualize = 3 
tree_to_visualize = rf_classifier.estimators_[tree_index_to_visualize]

dot_data = export_graphviz(
    tree_to_visualize,
    out_file=None,
    feature_names=feature_names,
    class_names=["Not Difficult", "Difficult"],
    filled=True,
    rounded=True,
    special_characters=True,
)

# Save and Display the tree
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree")  
graph.view()

# Predict the tax_problem values on the testing data
predictions = rf_classifier.predict(X_test)

# Calculate the evaluation metrics of the classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy:", accuracy)
print("f1_score:", f1)
print("precision:", precision)
print("recall:", recall)
print("Confusion matrix:", conf_matrix)

# Trained model 
trained_rf_model = rf_classifier

# save the order of the columns
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# Save the trained model
joblib.dump(trained_rf_model, 'trained_rf_model.pkl')

# Visulaization of different parameters

# 3D scatter plot
feature1 = 'dépenses'
feature2 = 'imposition_supplémentaire_évaluée'

X_feature1 = data[feature1]
X_feature2 = data[feature2]
predictions = rf_classifier.predict(data.drop(['difficulté_fiscale', 'type_de_donnée_encoded'], axis=1))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X_feature1, X_feature2, predictions, c=predictions, cmap='viridis')
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel("predictions")

cbar = fig.colorbar(sc)
cbar.set_label('tax_problem')

# Learning curve plot
plt.figure(figsize=(12, 6))
plt.title("Random Forest Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")

# Specify the estimator (Random Forest in this case)
estimator = rf_classifier

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")

plt.legend(loc="best")

# Create a bar chart to represent the importance of features
plt.figure(figsize=(8, 6))
feature=[ i.replace('_',' ') for i in feature_names]
feature=[ i.replace('encoded','') for i in feature]
print(feature)
plt.barh(feature, color='skyblue',width=feature_importance)
plt.xlabel('Relevance')
plt.title('Key Feature Relevance in Predicting Tax Issues')

# Display the chart
plt.show()


