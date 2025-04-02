from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model
model = joblib.load("depression_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="🎓 Student Depression Detection API")

# Define input data model
class StudentData(BaseModel):
    student_id: int
    gender: int
    age: float
    city: int
    profession: int
    academic_pressure: float
    work_pressure: float
    cgpa: float
    study_satisfaction: float
    job_satisfaction: float
    sleep_duration: int
    dietary_habits: int
    degree: int
    suicidal_thoughts: int
    work_study_hours: float
    financial_stress: int
    family_history: int

# API Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Depression Detection API!"}

# Prediction Endpoint
@app.post("/predict")
def predict_depression(data: StudentData):
    # Convert input data to NumPy array
    new_student_data = np.array([
        data.student_id, data.gender, data.age, data.city, data.profession,
        data.academic_pressure, data.work_pressure, data.cgpa, data.study_satisfaction,
        data.job_satisfaction, data.sleep_duration, data.dietary_habits, data.degree,
        data.suicidal_thoughts, data.work_study_hours, data.financial_stress, data.family_history
    ]).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(new_student_data)

    # Return result as JSON response
    result = "The student may be experiencing depression." if prediction[0] == 1 else "The student is unlikely to be experiencing depression."

    return {"prediction": result}
