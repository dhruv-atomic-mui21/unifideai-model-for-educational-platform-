import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationModel:
    def __init__(self, data):
        """
        Initialize the recommendation model with data.
        :param data: DataFrame containing 'student_id', 'resource_id', and 'resource_description'.
        """
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.resource_vectors = None

    def preprocess_data(self):
        """
        Preprocess the resource descriptions to create feature vectors.
        """
        self.resource_vectors = self.vectorizer.fit_transform(self.data['resource_description'])

    def recommend_resources(self, student_id, top_n=3):
        """
        Recommend resources for a given student based on their interactions.
        :param student_id: ID of the student to recommend resources for.
        :param top_n: Number of top recommendations to return.
        :return: List of recommended resource IDs.
        """
        # Get resources the student has interacted with
        student_resources = self.data[self.data['student_id'] == student_id]
        if student_resources.empty:
            print(f"No data found for student ID {student_id}")
            return []

        # Calculate similarity scores for all resources
        student_profile = np.asarray(self.resource_vectors[student_resources.index].mean(axis=0))
        similarity_scores = cosine_similarity(student_profile, self.resource_vectors).flatten()

        # Rank resources by similarity scores
        recommendations = self.data.copy()
        recommendations['similarity'] = similarity_scores
        recommendations = recommendations[~recommendations['resource_id'].isin(student_resources['resource_id'])]
        top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)

        return top_recommendations['resource_id'].tolist()
