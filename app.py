from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fpdf import FPDF
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['REPORT_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return "<h3 style='color:red;'>No file uploaded. Please upload a CSV file.</h3>"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"<h3 style='color:red;'>Error reading CSV: {str(e)}</h3>"

        if df.empty:
            return "<h3 style='color:red;'>Uploaded CSV is empty.</h3>"

        # Handle nulls
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0].to_dict()
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Detect target column (last column)
        try:
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]

            if y.dtype == 'O':
                y = pd.factorize(y)[0]

            X = pd.get_dummies(X)
            X.fillna(0, inplace=True)
        except Exception as e:
            return f"<h3 style='color:red;'>Error processing data: {str(e)}</h3>"

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        except Exception as e:
            return f"<h3 style='color:red;'>Train-test split failed: {str(e)}</h3>"

        # Train Random Forest (only one model to reduce memory usage)
        try:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
        except Exception as e:
            return f"<h3 style='color:red;'>Model training failed: {str(e)}</h3>"

        # Save model
        try:
            model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            return f"<h3 style='color:red;'>Model save failed: {str(e)}</h3>"

        # Generate PDF report
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Model: Random Forest", ln=True)
            pdf.multi_cell(0, 10, txt=f"Accuracy: {acc:.2f}\n\n{report}")
            pdf.output(os.path.join(app.config['REPORT_FOLDER'], 'report.pdf'))
        except Exception as e:
            return f"<h3 style='color:red;'>Report generation failed: {str(e)}</h3>"

        # Return dashboard
        return render_template('dashboard.html',
                               nulls=null_cols,
                               data=df.head().to_html(classes="data"),
                               target_col=target_col,
                               best_model="Random Forest",
                               acc=acc,
                               report=report,
                               model_url='/download/model',
                               report_url='/download/report',
                               img_url=None)

    except Exception as e:
        traceback.print_exc()
        return f"<h3 style='color:red;'>Unexpected error: {str(e)}</h3>"

@app.route('/download/<filetype>')
def download(filetype):
    try:
        if filetype == 'model':
            return send_file('models/best_model.pkl', as_attachment=True)
        elif filetype == 'report':
            return send_file('reports/report.pdf', as_attachment=True)
        else:
            return "<h3 style='color:red;'>Invalid file type requested.</h3>"
    except Exception as e:
        return f"<h3 style='color:red;'>Download failed: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
