from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from fpdf import FPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['REPORT_FOLDER'] = 'reports'

os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded."

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)

    # Handle nulls
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].to_dict()
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Auto-detect target (last column)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'O':
        y = pd.factorize(y)[0]

    X = pd.get_dummies(X)
    X.fillna(0, inplace=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC()
    }

    accuracies = {}
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        reports[name] = classification_report(y_test, y_pred)

    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_acc = accuracies[best_model_name]
    best_report = reports[best_model_name]

    # Save best model
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Save report PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Best Model: {best_model_name}", ln=True)
    pdf.multi_cell(0, 10, txt=f"Accuracy: {best_acc:.2f}\n\n{best_report}")
    pdf.output(os.path.join(app.config['REPORT_FOLDER'], 'report.pdf'))

    # Accuracy chart
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/model_accuracy.png')

    return render_template('dashboard.html',
                           nulls=null_cols,
                           data=df.head().to_html(classes="data"),
                           target_col=target_col,
                           best_model=best_model_name,
                           acc=best_acc,
                           report=best_report,
                           model_url='/download/model',
                           report_url='/download/report',
                           img_url='/static/model_accuracy.png')

@app.route('/download/<filetype>')
def download(filetype):
    if filetype == 'model':
        return send_file('models/best_model.pkl', as_attachment=True)
    elif filetype == 'report':
        return send_file('reports/report.pdf', as_attachment=True)
    return "Invalid file type."

if __name__ == '__main__':
    app.run(debug=True)
