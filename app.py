from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import openai
import requests
from sklearn.calibration import LabelEncoder  # Import the OpenAI library

app = Flask(__name__, template_folder='templates')

api_key = 'Enter your API key'


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    recommendations = None
    if request.method == 'POST':
        # Get user input from the form
        industrie = request.form['industrie']
        chiffre_affaires = int(request.form['chiffre_affaires'])
        depenses = int(request.form['dépenses'])
        categorie_actifs = request.form['catégorie_d\'actifs']
        valeur_actifs = int(request.form['valeur_d\'actifs'])
        age_actifs = int(request.form['âge_d\'actifs'])
        imposition_eval = int(
            request.form['imposition_supplémentaire_évaluée'])
        creances_irrecouvrables = int(request.form['créances_irrécouvrables'])

        # Load the trained model
        trained_rf_model = joblib.load('trained_rf_model.pkl')

        # Load the column order of the training data
        model_columns = joblib.load('model_columns.pkl')

        # Create a DataFrame from user input
        user_input = pd.DataFrame({
            'industrie': [industrie],
            'chiffre_affaires': [chiffre_affaires],
            'dépenses': [depenses],
            'catégorie_d\'actifs': [categorie_actifs],
            'valeur_d\'actifs': [valeur_actifs],
            'âge_d\'actifs': [age_actifs],
            'imposition_supplémentaire_évaluée': [imposition_eval],
            'créances_irrécouvrables': [creances_irrecouvrables]
        })

        # Preprocess user input
        label_encoder = LabelEncoder()

        user_input['industrie_encoded'] = label_encoder.fit_transform(user_input['industrie'])
        user_input.drop(['industrie'], axis=1, inplace=True)

        user_input['catégorie_d\'actifs_encoded'] = label_encoder.fit_transform(user_input['catégorie_d\'actifs'])
        user_input.drop(['catégorie_d\'actifs'], axis=1, inplace=True)

        # Reorder columns to match the model's expected input
        user_input = user_input[model_columns]

        # Predict using the trained model
        prediction = trained_rf_model.predict(user_input)

        if prediction[0] == 0:
            result = "No tax problem is predicted for the company."
        else:
            result = "A tax problem is predicted for the company."

        # Prepare the input prompt for GPT-4
        input_prompt = "Tax Problem Description:\n"
        input_prompt += f"- Industry: {industrie}\n"
        input_prompt += f"- Chiffre d'affaires: {chiffre_affaires}\n"
        input_prompt += f"- Dépenses: {depenses}\n"
        input_prompt += f"- Catégorie d'actifs: {categorie_actifs}\n"
        input_prompt += f"- Valeur d'actifs: {valeur_actifs}\n"
        input_prompt += f"- Âge d'actifs: {age_actifs}\n"
        input_prompt += f"- Imposition supplémentaire évaluée: {imposition_eval}\n"
        input_prompt += f"- Créances irrécouvrables: {creances_irrecouvrables}\n"
        input_prompt += f"- Probleme fiscale: {result}\n"
        input_prompt += "Apart from consulting an expert, what can the company do to avoid this tax problem?\n\nGive answer in a single paragraph"

        messages = [{"role": "user", "content": input_prompt}]
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "gpt-4", "messages": messages}
        response = requests.post(url, headers=headers, json=data).json()

        

        # Extract the generated recommendation from the GPT-3 response
        recommendations = response['choices'][0]['message']['content']
        #recommendations = recommendations.replace(recommendations[0], "", 1)
        return redirect(url_for('show_result', prediction=result, recommendations=recommendations))

    return render_template('index.html')

   

@app.route('/result')

def show_result():
    result = request.args.get('prediction')
    recommendations = request.args.get('recommendations') 
    return render_template('result.html', result=result, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
