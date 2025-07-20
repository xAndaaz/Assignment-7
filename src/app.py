import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('model/penguin_model.pkl')
model_columns = joblib.load('model/model_columns.pkl')
penguins = sns.load_dataset("penguins").dropna()

# Page setup
st.set_page_config(page_title="Penguin Predictor", layout="wide")
st.title("🐧 Penguin Species Predictor")

# --- Sidebar ---
st.sidebar.header("Input Features")

def user_input_features():
    """Defines sidebar widgets and returns user inputs."""
    island = st.sidebar.selectbox("Island", penguins['island'].unique())
    sex = st.sidebar.selectbox("Sex", penguins['sex'].unique())
    bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.1)
    flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 200.0)
    body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4200.0)

    data = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island_Dream": island == "Dream",
        "island_Torgersen": island == "Torgersen",
        "sex_male": sex == "Male",
    }
    
    features = pd.DataFrame(data, index=[0])
    
    # Reorder columns to match model's expectations
    input_df = pd.DataFrame(columns=model_columns)
    input_df = pd.concat([input_df, features])
    input_df.fillna(0, inplace=True)
    
    return input_df[model_columns]

input_df = user_input_features()

# --- Prediction Display ---
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.metric("Predicted Species", prediction[0])
    st.write("Prediction Confidence:")
    st.write(pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probability"]))

with pred_col2:
    image_map = {
        "Adelie": "image/adelie.jpeg",
        "Chinstrap": "image/chinstrap.png",
        "Gentoo": "image/gentoo.jpg"
    }
    image_path = image_map.get(prediction[0])
    if image_path:
        st.image(image_path, width=300)

# --- Visual Analysis ---
st.header("Visual Analysis")

# Scatter plots
st.subheader("How Your Input Compares")
col1, col2 = st.columns(2)

with col1:
    st.markdown("Bill Length vs. Bill Depth")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", style="species", ax=ax1)
    ax1.scatter(input_df['bill_length_mm'], input_df['bill_depth_mm'], marker='X', s=200, c='red', label='Your Input')
    st.pyplot(fig1)

with col2:
    st.markdown("Flipper Length vs. Body Mass")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g", hue="species", style="species", ax=ax2)
    ax2.scatter(input_df['flipper_length_mm'], input_df['body_mass_g'], marker='X', s=200, c='red', label='Your Input')
    st.pyplot(fig2)

# Model insights
st.subheader("Model Insights")
col3, col4 = st.columns(2)

with col3:
    st.markdown("Feature Importance")
    feature_imp = pd.DataFrame(model.feature_importances_, index=model_columns, columns=['Importance']).sort_values('Importance', ascending=False)
    st.bar_chart(feature_imp)

with col4:
    st.markdown("Feature Correlation")
    numeric_penguins = penguins.select_dtypes(include=['float64', 'int64'])
    corr = numeric_penguins.corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# Data overview
with st.expander("View Raw Dataset"):
    st.write("The Palmer Penguins dataset, used for training the model.")
    st.dataframe(penguins.head())
    st.write("Species Distribution:")
    species_counts = penguins['species'].value_counts()
    st.bar_chart(species_counts)
