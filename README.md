📌 Title: Student Score Prediction

📌 Objective:
The main goal of this project was to develop a regression-based machine learning model to predict students' exam scores using various lifestyle and academic factors.
The models explored how study habits, sleep, attendance, exercise, and mental health relate to academic performance, using both Linear Regression and Polynomial Regression (Degree 2).

📂 Dataset:
The dataset used for this analysis was sourced from Kaggle’s “Student Academic Performance Analysis”.
The following variables were selected as features:

study_hours_per_day

attendance_percentage

sleep_hours

exercise_frequency

mental_health_rating

exam_score (target)

🧹 Data Exploration and Visualization:
I started by:

Checking for missing values and cleaning the dataset.

Plotting a correlation heatmap to identify the most predictive features.

Visualizing the actual vs. predicted scores, along with a reference line (y = x) and regression fit line to compare model performance visually.

🤖 Model Development:

🔍 Linear Regression:
A standard linear regression model was trained using an 80:20 train-test split.

R² Score: 0.86

Mean Squared Error (MSE): 36.40

The model successfully explained 86% of the variance in student exam scores. 💪

🔍 Polynomial Regression (Degree 2):
To capture possible non-linear relationships, I applied a second model using Polynomial Regression (degree 2).

R² Score: 0.86

MSE: 35.65

This model slightly reduced the prediction error but didn’t significantly improve overall performance. 🔁

🧠 Conclusion:
📊 The comparison between the two models reveals that both Linear and Polynomial Regression (degree 2) performed almost equally in terms of prediction accuracy. The Linear Regression model achieved an R² Score of 0.86 and an MSE of 36.40, while the Polynomial Regression model had an R² Score of 0.86 and a slightly lower MSE of 35.65.
Despite the small difference in error, the linear model remains the simpler and more efficient choice for this task. It captures the relationship between study habits and exam scores effectively without added complexity. Therefore, Linear Regression is considered the better model for predicting exam scores based on the available features.
This project helped reinforce the power of simple linear models and the importance of feature selection, visualization, and evaluation metrics in machine learning workflows.





#MachineLearning #Regression #Python #DataScience #StudentProject #AI #ScikitLearn #Visualization #Kaggle #LinkedInLearning #PredictiveAnalytics
