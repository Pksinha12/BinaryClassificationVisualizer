import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Set custom CSS styles
st.markdown("""
<style>
body {
    color: #1E2833;
    background-color: #F2F4F6;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Binary Classification Visualizer")
    # Add a file uploader widget in the sidebar
    st.sidebar.subheader("Choose CSV File Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"])

    if uploaded_file is not None:
        st.title(f"{uploaded_file.name}")
    else:
        st.title("Mushroom.csv")

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, value=0.3, step=0.05)
    random_state = st.sidebar.number_input("Random State", 0, 1000, value=0, step=1)

    #@st.cache_data(allow_output_mutation=True)
    def load_data():
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv("mushrooms.csv")
        return data

    def preprocess_data(df):
        # Handle missing values (if any)
        df.fillna(0, inplace=True)

        # Encode categorical features (if any)
        label_encoder = LabelEncoder()
        for column in df.select_dtypes(include='object'):
            df[column] = label_encoder.fit_transform(df[column])

        return df

    def get_target_column(df):
        st.sidebar.subheader("Select Target Column")
        target_column = st.sidebar.selectbox("Target Column", df.columns)
        return target_column

    @st.cache_data()
    def split(df, target_column):
        y = df[target_column]
        x = df.drop(columns=[target_column])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def plot_confusion_matrix(model, x_test, y_test, display_labels):
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=display_labels, yticklabels=display_labels,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.gca().invert_yaxis()

    def plot_metrics(metrics_list, model, target_column):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            y_pred_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            st.pyplot(fig)

    df = load_data()
    df = preprocess_data(df)
    target_column = get_target_column(df)
    class_names = df[target_column].unique().tolist()

    x_train, x_test, y_train, y_test = split(df, target_column)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        # choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy_score(y_test, y_pred).round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            plot_metrics(metrics, model, target_column)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy_score(y_test, y_pred).round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            plot_metrics(metrics, model, target_column)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators_RF')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth_RF')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False),
                                     key='bootstrap_RF')  # Convert to boolean
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                           n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy_score(y_test, y_pred).round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            plot_metrics(metrics, model, target_column)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Data Set")
        st.write(df)

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
