from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy import sparse

from flask import request, jsonify
import json

app = Flask(__name__)

df = pd.read_csv('disease_symptoms.csv')
symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
df['symptoms'] = df[symptom_cols].fillna('').apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.lower()
df['symptoms'] = df['symptoms'].str.replace('_', ' ')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['symptoms'].lower().replace('_', ' ')
    if not user_input.strip():
        return render_template('index.html', results=[], warning="Please enter symptoms.")
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    vectors = sparse.load_npz('vectors.npz')

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors)[0]
    df['similarity'] = similarities
    results = list(
        df.sort_values(by='similarity', ascending=False)
        .drop_duplicates(subset='Disease')
        .head(5)[['Disease', 'similarity']]
        .itertuples(index=False, name=None)
    )

    desc_df = pd.read_csv('description.csv')    
    diet_df = pd.read_csv('diets.csv')           
    prec_df = pd.read_csv('precautions.csv')      
    ans = []
    for idx in range(0,len(results)):
        desc=desc_df[desc_df['Disease']==results[idx][0]]['Description']
        desc = " ".join([w for w in desc])
        ans.append((results[idx][0],desc))
          

    diet = []
   
    for disease, _ in results:
        diet_list = diet_df[diet_df['Disease'] == disease]['Diet'].tolist()
        if diet_list:
            diet.append((disease, diet_list[0]))
        else:
            diet.append((disease, "No diet information available."))
  
        precautions = []
        for disease, _ in results:
            row = prec_df[prec_df['Disease'] == disease]
            if not row.empty:
        # Extract all precaution columns
                prec_list = row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].tolist()
                precautions.append((disease, prec_list))
            else:
                precautions.append((disease, ["No precautions available."]))

    return render_template('index.html', predicted_disease=results,disease_description=ans,diet_list=diet ,precautions_list=precautions,warning=None)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').lower().replace('_', ' ')

    if not user_input.strip():
        return "Please describe your symptoms."

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    vectors = sparse.load_npz('vectors.npz')

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors)[0]
    df['similarity'] = similarities
    results = list(
        df.sort_values(by='similarity', ascending=False)
        .drop_duplicates(subset='Disease')
        .head(3)[['Disease', 'similarity']]
        .itertuples(index=False, name=None)
    )

    if not results:
        return "Sorry, I couldn't detect any disease based on the symptoms."

    response = "Based on your symptoms, you might have: "
    response += ", ".join([f"{d} ({round(s*100, 1)}%)" for d, s in results])
    return response




if __name__ == '__main__':
    app.run(debug=True)












