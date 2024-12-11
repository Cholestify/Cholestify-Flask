import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Path to your SavedModel
MODEL_PATH = 'C:\\Users\\schai\\Downloads\\deploy flask2\\bangkitmodel'

# Load the SavedModel using the signatures
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Load food data
food_data = pd.read_csv("C:\\Users\\schai\\Downloads\\deploy flask2\\data\\food_data.csv")
food_features = tf.convert_to_tensor(
    food_data[['Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Cholesterol']].values,
    dtype=tf.float32
)

# Menangani permintaan POST untuk prediksi
@app.route('/recommend', methods=['POST'])
def recommend_food():
    try:
        # Parse the input JSON
        user_nutrition = request.json.get("user_nutrition")

        if not user_nutrition:
            return jsonify({"error": "user_nutrition is required"}), 400

        # Convert to tensor
        user_nutrition = tf.convert_to_tensor([user_nutrition], dtype=tf.float32)

        # Generate embeddings
        user_embedding = model.query_model(user_nutrition)
        food_embeddings = model.candidate_model(food_features)

        # Compute similarity scores
        dot_product = tf.linalg.matmul(user_embedding, food_embeddings, transpose_b=True)
        user_norm = tf.norm(user_embedding, axis=1, keepdims=True)
        food_norm = tf.norm(food_embeddings, axis=1, keepdims=True)
        similarity_scores = dot_product / (user_norm * tf.transpose(food_norm) + 1e-9)

        # Get top 5 recommendations
        top_k_indices = tf.argsort(similarity_scores, direction="DESCENDING")[0][:3]
        recommended_food_names = food_data['food'].iloc[top_k_indices.numpy()]

        return jsonify({"Food Recommendations": recommended_food_names.to_list()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

