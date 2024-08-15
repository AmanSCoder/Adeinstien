from flask import Flask, request, jsonify
import pickle

# Load the model and vectorizer
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

def print_cluster(i):
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]
    return cluster_terms

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    product_description = data['product_description']
    
    Y = vectorizer.transform([product_description])
    prediction = model.predict(Y)
    
    recommendations = print_cluster(prediction[0])
    
    return jsonify({
        'cluster': int(prediction[0]),
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
