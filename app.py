import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Page Configuration
st.set_page_config(
    page_title="Employee Income Analytics",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Employee Income Analytics & Prediction")
st.write(
    "Analyze employee data and predict whether annual income "
    "falls above or below $50K."
)

#Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(
        "model/salary_prediction_model.pkl"
    )
model = load_model()

#Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/adult3.csv")

df = load_data()

#Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Prediction",
        "📊 Analytics Dashboard",
        "📁 Batch Prediction"
    ]
)

#Individual Prediction Page
if page == "🔮 Prediction":

    st.header("🔮 Employee Income Prediction")

    st.write(
        "Enter the employee details below to predict "
        "the income category."
    )

    col1, col2 = st.columns(2)

    # -----------------------------
    # Personal Information
    # -----------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=90,
            value=30,
            key="age_input"
        )

        workclass = st.selectbox(
            "Workclass",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ],
            key="workclass_input"
        )

        education = st.selectbox(
            "Education",
            [
                "Bachelors",
                "Some-college",
                "11th",
                "HS-grad",
                "Masters",
                "9th",
                "Doctorate",
                "Assoc-acdm",
                "Assoc-voc",
                "7th-8th",
                "12th",
                "10th",
                "1st-4th",
                "5th-6th",
                "Prof-school",
                "Preschool"
            ],
            key="education_input"
        )

        educational_num = st.number_input(
            "Education Number",
            min_value=1,
            max_value=16,
            value=10,
            key="educational_num_input"
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse"
            ],
            key="marital_status_input"
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Adm-clerical",
                "Farming-fishing",
                "Transport-moving",
                "Priv-house-serv",
                "Protective-serv",
                "Armed-Forces"
            ],
            key="occupation_input"
        )

        relationship = st.selectbox(
            "Relationship",
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried"
            ],
            key="relationship_input"
        )

    # -----------------------------
    # Employment / Financial Info
    # -----------------------------

    with col2:

        race = st.selectbox(
            "Race",
            [
                "White",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other",
                "Black"
            ],
            key="race_input"
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ],
            key="gender_input"
        )

        fnlwgt = st.number_input(
            "Final Weight (fnlwgt)",
            min_value=0,
            value=100000,
            key="fnlwgt_input"
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            value=0,
            key="capital_gain_input"
        )

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            value=0,
            key="capital_loss_input"
        )

        hours_per_week = st.number_input(
            "Hours per Week",
            min_value=1,
            max_value=100,
            value=40,
            key="hours_per_week_input"
        )

        native_country = st.selectbox(
            "Native Country",
            sorted(
                df["native-country"]
                .dropna()
                .unique()
                .tolist()
            ),
            key="native_country_input"
        )

    st.divider()

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button(
        "🔮 Predict Income",
        type="primary",
        key="predict_button"
    ):

        input_data = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "fnlwgt": [fnlwgt],
            "education": [education],
            "educational-num": [educational_num],
            "marital-status": [marital_status],
            "occupation": [occupation],
            "relationship": [relationship],
            "race": [race],
            "gender": [gender],
            "capital-gain": [capital_gain],
            "capital-loss": [capital_loss],
            "hours-per-week": [hours_per_week],
            "native-country": [native_country]
        })

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        st.success(
            f"Predicted Income: {prediction}"
        )

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Income Class": classes,
            "Probability": probabilities
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        st.bar_chart(
            probability_df.set_index("Income Class")
        )

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.header("📊 Employee Income Analytics")

    st.write(
        "Explore patterns and relationships in the employee "
        "income dataset."
    )

    # --------------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------------

    st.subheader("📌 Dataset Overview")

    total_records = len(df)
    total_features = len(df.columns) - 1
    income_classes = df["income"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Input Features",
        total_features
    )

    col3.metric(
        "Income Classes",
        income_classes
    )

    st.divider()

    # --------------------------------------------------------
    # Income Distribution
    # --------------------------------------------------------

    st.subheader("💰 Income Distribution")

    income_counts = df["income"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=income_counts.index,
        y=income_counts.values,
        ax=ax
    )

    ax.set_xlabel("Income Category")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Employee Income Distribution")

    st.pyplot(fig)

    # --------------------------------------------------------
    # Income vs Education
    # --------------------------------------------------------

    st.subheader("🎓 Income by Education")

    education_income = pd.crosstab(
        df["education"],
        df["income"]
    )

    education_income = education_income.sort_values(
        by=">50K",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    education_income.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Education")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Income Distribution Across Education Levels")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Income vs Age
    # --------------------------------------------------------

    st.subheader("👤 Income vs Age")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="income",
        y="age",
        ax=ax
    )

    ax.set_xlabel("Income Category")
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution Across Income Categories")

    st.pyplot(fig)

    # --------------------------------------------------------
    # Income vs Hours Worked
    # --------------------------------------------------------

    st.subheader("⏰ Income vs Working Hours")

    hours_income = (
        df.groupby("income")["hours-per-week"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=hours_income,
        x="income",
        y="hours-per-week",
        ax=ax
    )

    ax.set_xlabel("Income Category")
    ax.set_ylabel("Average Hours per Week")
    ax.set_title("Average Working Hours by Income Category")

    st.pyplot(fig)

    # --------------------------------------------------------
    # Income vs Occupation
    # --------------------------------------------------------

    st.subheader("💼 Income by Occupation")

    occupation_income = pd.crosstab(
        df["occupation"],
        df["income"]
    )

    occupation_income = occupation_income.sort_values(
        by=">50K",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    occupation_income.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Occupation")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Top Occupations by Income Category")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    st.pyplot(fig)

# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.header("📁 Batch Income Prediction")

    st.write(
        "Upload a CSV file containing employee information "
        "to generate income predictions for multiple records."
    )

    st.info(
        "The uploaded CSV must contain the same 14 input "
        "features used during model training."
    )

    uploaded_file = st.file_uploader(
        "Upload Employee CSV",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded_file is not None:

        batch_data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")

        st.dataframe(
            batch_data.head(),
            use_container_width=True
        )

        # Required features
        required_features = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "educational-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"
        ]

        missing_features = [
            feature
            for feature in required_features
            if feature not in batch_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded CSV is missing the following "
                f"columns: {missing_features}"
            )

        else:

            if st.button(
                "🚀 Generate Predictions",
                type="primary",
                key="batch_predict_button"
            ):

                predictions = model.predict(
                    batch_data[required_features]
                )

                probabilities = model.predict_proba(
                    batch_data[required_features]
                )

                result = batch_data.copy()

                result["Predicted Income"] = predictions

                # Find probability of >50K
                class_index = list(
                    model.classes_
                ).index(">50K")

                result["Probability >50K"] = (
                    probabilities[:, class_index] * 100
                ).round(2)

                st.success(
                    f"Successfully generated predictions "
                    f"for {len(result):,} employees."
                )

                st.subheader("📊 Prediction Results")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Download results
                # ------------------------------------------------

                csv = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv,
                    file_name="employee_income_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )