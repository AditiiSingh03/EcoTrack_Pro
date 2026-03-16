# EcoTrack Pro – AI-Powered Carbon Footprint Analytics Platform

EcoTrack Pro is an interactive sustainability analytics platform that estimates annual carbon footprint from user lifestyle inputs and provides explainable, AI-assisted recommendations for reducing environmental impact.

The application combines machine learning, SHAP explainability, AI-generated sustainability advice, PDF reporting, and interactive dashboards in a single Streamlit app.

---

## Project Overview

Carbon footprint is influenced by multiple lifestyle factors such as transportation, electricity usage, and consumption habits.

EcoTrack Pro helps users:

- Estimate annual carbon emissions from lifestyle inputs
- Visualize their environmental impact through charts and gauges
- Understand key drivers behind predictions using SHAP explainability
- Simulate possible emission reductions
- Receive AI-generated sustainability strategies
- Download a personalized carbon footprint report
- Interact with an AI chatbot and image-based sustainability scanner

---

## Key Features

### Carbon Footprint Prediction
Predicts a user's estimated annual carbon footprint based on structured lifestyle and consumption inputs.

### Model Performance Dashboard
Displays core evaluation metrics such as R², RMSE, and MAE for the trained machine learning model.

### Emission Breakdown Visualization
Shows estimated carbon contribution across major categories such as:

- Transport
- Electricity
- Lifestyle

### SHAP Explainability
Explains why a specific prediction is high or low using feature-level importance for that individual user input.

### What-If Reduction Simulator
Allows users to simulate reductions in travel and electricity-related usage and estimate the resulting decrease in emissions.

### AI Sustainability Advice
Generates personalized action recommendations using Gemini API.

### PDF Report Generation
Creates a downloadable report containing:

- User inputs
- Carbon footprint estimate
- SHAP logic breakdown
- AI strategy recommendations

### Vision AI Scanner
Lets users upload an image and receive AI-based sustainability analysis of the detected object.

### Expert Chatbot
Provides an AI-powered chat interface for sustainability-related questions and guidance.

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Matplotlib
- SHAP
- Gemini API
- FPDF
- Pillow

---

## Project Structure

```text
EcoTrack-Pro
│
├── app.py
├── requirements.txt
├── background.jpg
├── check_api.py
├── config.toml
├── users.json
└── README.mdec

---

## How to Run the Project

Clone the repository
git clone https://github.com/AditiiSingh03/EcoTrack-Pro.git
Navigate to the project folder
cd EcoTrack-Pro
Install dependencies
pip install -r requirements.txt
Run the Streamlit app
streamlit run app.py
The application will start locally in your browser.

---

## Secrets Configuration

This project uses the Gemini API to generate AI-based sustainability recommendations, power the chatbot assistant, and analyze uploaded images.
For local development, create a file:
.streamlit/secrets.toml
Add the following configuration:
GEMINI_API_KEY = "your_api_key_here"
For Streamlit Cloud deployment:
1. Open your deployed app dashboard.
2. Go to **Settings → Secrets**.
3. Add the same key.

---

## Model Handling

The trained machine learning model is not stored directly inside the repository because GitHub restricts large files.

Instead, the application automatically downloads the trained model from a GitHub Release when the app starts.

This approach keeps the repository lightweight and allows the model to be updated independently without modifying the application code.

---

## Future Improvements

Possible future extensions of the platform include:

- Category-level carbon prediction using specialized sub-models
- Historical emission tracking for each user
- Personalized weekly sustainability goals
- Regional and global emission comparison dashboards
- Integration with real-world emission factor databases
- Database-backed user authentication system
- Mobile-friendly UI improvements

---

## Author

**Aditi Singh**  
B.Tech Computer Science & Information Technology  
KIET Group of Institutions, Ghaziabad  

GitHub:  
https://github.com/AditiiSingh03

LinkedIn:  
https://www.linkedin.com/in/aditi-singh-991b22288/
