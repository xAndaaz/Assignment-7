# 🐧 Penguin Species Prediction App

### An Interactive Machine Learning Web Application

This project demonstrates how to deploy a trained machine learning model as a user-friendly and interactive web application using Streamlit. The goal is to bridge the gap between a functional model and an accessible end-product, allowing users to not only get predictions but also to understand the model's behavior and the underlying data.

The app uses a Random Forest Classifier trained on the classic Palmer Penguins dataset to predict a penguin's species (Adelie, Chinstrap, or Gentoo) based on its physical measurements.

---

## Key Features

- **Real-Time Prediction:** An intuitive sidebar allows users to input a penguin's measurements. The model instantly predicts the species and displays the result, complete with an image of the penguin and the model's confidence level.

- **Rich Visual Analysis:** To go beyond simple predictions, the app includes a suite of visualizations designed to provide deeper insights:
  - **Input vs. Data Distribution:** Scatter plots show exactly where the user's custom input falls in relation to the original dataset, providing immediate context.
  - **Model Insights:** A feature importance chart reveals which characteristics the model weighs most heavily, and a correlation heatmap shows how different measurements relate to each other.
  - **Dataset Overview:** An expandable section provides a look at the raw training data and the distribution of species, offering transparency about the data used.

- **Polished User Interface:** The entire application is built with a clean, professional layout, ensuring a smooth and engaging user experience.

---

## Quick Start

To run this application locally, clone the repository and use the following commands:

```bash
# Install dependencies, train the model, and run the app
pip install -r requirements.txt
python src/main.py
streamlit run src/app.py
```
