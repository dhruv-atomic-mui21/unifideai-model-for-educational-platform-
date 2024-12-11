import pandas as pd
from model import RecommendationModel

# Sample dataset
data = pd.DataFrame({
    'student_id': [1, 1, 2, 2, 3],
    'resource_id': [101, 102, 103, 101, 104],
    'resource_description': [
        'Introduction to Python programming',
        'Advanced Python concepts',
        'Data Science with Python',
        'Introduction to Python programming',
        'Machine learning basics'
    ]
})

# Initialize the recommendation model
recommendation_model = RecommendationModel(data)

# Preprocess data
recommendation_model.preprocess_data()

# Get recommendations for a student
student_id = 1
recommendations = recommendation_model.recommend_resources(student_id)
print(f"Recommended resources for student {student_id}: {recommendations}")
