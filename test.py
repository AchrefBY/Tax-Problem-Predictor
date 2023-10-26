import pandas as pd
import joblib
from sklearn.calibration import LabelEncoder

label_encoder = LabelEncoder()

# Load the trained Random Forest model
rf_model = joblib.load('trained_rf_model.pkl')

# Load the column order of the training data
model_columns = joblib.load('model_columns.pkl')

# Gather user input (replace these values with actual user input)
user_input = {
    'industrie': 'Construction',
    'chiffre_affaires': 32000000,
    'dépenses': 23000000,
    'catégorie_d\'actifs': 'Amortissable',
    'valeur_d\'actifs': 12300500,
    'âge_d\'actifs': 20,
    'imposition_supplémentaire_évaluée': 420000,
    'créances_irrécouvrables': 0,
}

# Convert user input into a DataFrame
user_df = pd.DataFrame(user_input, index=[0])

# Preprocess user input

user_df['industrie_encoded'] = label_encoder.fit_transform(user_df['industrie'])
user_df.drop(['industrie'], axis=1, inplace=True)

user_df['catégorie_d\'actifs_encoded'] = label_encoder.fit_transform(user_df['catégorie_d\'actifs'])
user_df.drop(['catégorie_d\'actifs'], axis=1, inplace=True)

# Reorder user input columns to match the model's expected column order
user_df = user_df[model_columns]

# Predict using the trained model
user_prediction = rf_model.predict(user_df)

# Display the prediction result
if user_prediction[0] == 0:
    print("No tax problem is predicted for the company.")
else:
    print("A tax problem is predicted for the company.")
