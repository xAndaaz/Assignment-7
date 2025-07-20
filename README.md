# 🐧 Penguin Species Prediction App

An interactive web app built with Streamlit to deploy a machine learning model. This application predicts the species of a penguin from the Palmer Penguins dataset using a trained Random Forest Classifier and provides rich, interactive visualizations to explore the data and model behavior.

## Features

- **Interactive Sidebar:** Users can input custom values for a penguin's physical characteristics.
- **Instant Prediction:** The model predicts the species and displays the result with the penguin's image and the model's confidence score.
- **Dynamic Visualizations:** The app includes several plots to help users understand the data:
  - Scatter plots showing the user's input against the overall data distribution.
  - A bar chart of the model's feature importances.
  - A correlation heatmap of the dataset's numerical features.
- **Dataset Overview:** An expandable section shows the raw data and the distribution of penguin species.

## Project Structure

```
.
├── image/
│   ├── adelie.jpeg
│   ├��─ chinstrap.png
│   └── gentoo.jpg
├── model/
│   ├── penguin_model.pkl
│   └── model_columns.pkl
├── src/
│   ├── main.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup and Usage

### 1. Installation

Clone this repository and install the required packages. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

Before running the app for the first time, train the model by running the training script from the project's root directory:

```bash
python src/main.py
```

This populates the `model/` directory with the trained model file.

### 3. Run the Streamlit App

Launch the web application with the following command:

```bash
streamlit run src/app.py
```
This will open the application in your web browser.
