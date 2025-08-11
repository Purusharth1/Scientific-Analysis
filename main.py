from flask import Flask,render_template,request,session,redirect,url_for,flash,jsonify,send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user,logout_user,login_manager,LoginManager
from flask_login import login_required , current_user
from functools import wraps
from flask import abort
import numpy as np

app = Flask(__name__)

app.secret_key='Aditya'

login_manager = LoginManager(app)
login_manager.login_view = 'Login'

@login_manager.user_loader
def load_user(user_id):
    return User_table.query.get(int(user_id))


app.config['SQLALCHEMY_DATABASE_URI']='postgresql://postgres:Aditya164*@localhost:5432/Scientific_Analysis'
db=SQLAlchemy(app)


class User_table(UserMixin,db.Model):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(50))
    email = db.Column(db.String(50), unique = True)
    password = db.Column(db.String(1000))


@app.route("/")
# @login_required
def index():
    return render_template('homepage.html')

@app.route('/signup',methods =['POST','GET'])
def Signup():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user = User_table.query.filter_by(email=email).first()
        if user:
            flash("Email Already Exist","warning")
            return render_template('/signup.html')
        
        
        print(f"Username captured: {name}")  # Debugging line
        new_user = User_table(name = name , email = email , password = password)
        db.session.add(new_user)
        db.session.commit()
        with db.engine.connect() as conn:
            new_user = conn.execute(text(f"INSERT INTO User_table(name,email,password) VALUES ('{name}','{email}','{password}');"))
            conn.commit()

        flash("Signup Success Please Login","success")
        return render_template('login_signup.html')

@app.route('/login', methods = ['POST','GET'])
def Login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')

        
        user = User_table.query.filter_by(email = email).first()
        session['name'] = user.name
        if user and (user.password == password):
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('Login'))

    return render_template('login_signup.html')

@app.route("/logout")
@login_required
def Logout():
    logout_user()
    flash("Logout Successfull","warning")
    return redirect(url_for('Login'))

@app.route("/faqs")
def faq():
    return render_template('faq.html')







# For uploading file


import pandas as pd
import os

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

data_frame = pd.DataFrame()

@app.route('/upload')
def upload_index():
    return render_template('upload.html')

@app.route('/submit', methods=['POST'])
def submit():
    global data_frame

    # Capture data from the form
    columns = request.form.to_dict(flat=False)

    # Check if any columns were submitted
    if not columns or not any(columns.values()):  # Check if columns are empty or if no values are provided
        return jsonify({"message": "No data submitted."}), 400  # Return a bad request error

    # Calculate the number of rows by taking the length of the first column's data
    num_rows = len(columns[f'col1[]'])

    # Create a DataFrame using the submitted data
    data = {f'Column {i + 1}': [columns[f'col{i + 1}[]'][j] for j in range(num_rows)] for i in range(len(columns))}
    
    data_frame = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    data_frame.to_csv(csv_file_path, index=False)

    return jsonify({"message": "File submitted successfully!"})  # Return success message



@app.route('/upload', methods=['POST'])
def upload():
    global data_frame

    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_index'))
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Read CSV in chunks and fill missing values
    chunks = pd.read_csv(file_path, chunksize=10000)
    data_frame = pd.concat(chunk.fillna('') for chunk in chunks)  # Fill NaN with empty string

    return redirect(url_for('upload_index'))

@app.route('/get_data', methods=['GET'])
def get_data():
    global data_frame

    if data_frame.empty:
        return jsonify(columns=[], rows=[])

    data = {
        "columns": data_frame.columns.tolist(),
        "rows": data_frame.head(100).values.tolist()  # Only send the first 100 rows
    }
    return jsonify(data)



# for manually entering the data


@app.route('/manual')
def manually_index():
    return render_template('manually_data.html')


@app.route('/submit_manual', methods=['POST'])
def submit_manual():
    global data_frame

    # Capture data from the manual entry form
    columns = request.form.to_dict(flat=False)

    # Check if any columns were submitted
    if not columns or not any(columns.values()):  # Check if columns are empty or if no values are provided
        return jsonify({"message": "No data submitted."}), 400  # Return a bad request error

    # Calculate the number of rows by taking the length of the first column's data
    num_rows = len(columns[f'col1[]'])

    # Create a DataFrame using the submitted data
    data = {f'Column {i + 1}': [columns[f'col{i + 1}[]'][j] for j in range(num_rows)] for i in range(len(columns))}
    
    data_frame = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    data_frame.to_csv(csv_file_path, index=False)

    return jsonify({"message": "Data submitted successfully!"})  # Return success message










#plotting

  # A list of distinct colors

@app.route('/plot')
def plot_index():
    return render_template('plot.html')

@app.route('/gett_data', methods=['GET'])
def gett_data():
    # Return the columns of the DataFrame to populate dropdowns
    return jsonify({"columns": data_frame.columns.tolist()})

@app.route('/update_graph', methods=['POST'])
def update_graph():
    global data_frame

    print("Request received for graph update!")

    # Retrieve user inputs from the request
    x_feature = request.json.get('x_feature')
    y_feature = request.json.get('y_feature')
    z_feature = request.json.get('z_feature', None)  # Optional third feature
    graph_type = request.json.get('graph_type', 'scatter').lower()
    graph_subtype = request.json.get('graph_subtype', '')  # New graph subtype
    color = request.json.get('color', '#007bff')
    marker_size = request.json.get('marker_size', 10)
    show_grid = request.json.get('show_grid', True)

    print(f"X Feature: {x_feature}, Y Feature: {y_feature}, Z Feature: {z_feature}, Graph Type: {graph_type}, Graph Subtype: {graph_subtype}")

    # Validate features based on graph type
    if graph_type != 'heatmap':
        if x_feature not in data_frame.columns or y_feature not in data_frame.columns:
            return jsonify({"message": "Invalid X or Y feature selected."}), 400

    if z_feature and z_feature not in data_frame.columns:
        return jsonify({"message": "Invalid Z feature selected."}), 400

    response_data = {}  # Initialize the response data

    try:
        if graph_type == 'pie':
            # For pie charts, y_feature is typically categorical
            values = data_frame[y_feature].value_counts().tolist()
            labels = data_frame[y_feature].value_counts().index.tolist()
            # Optionally, define colors based on categories
            colors = [color] * len(labels)  # Simplistic: same color for all slices
            response_data['values'] = values
            response_data['labels'] = labels
            import plotly.express as px
            colors = px.colors.qualitative.Plotly
            response_data['colors'] = colors[:len(labels)]
            response_data['title'] = f'Pie Chart of {y_feature}'

        elif graph_type == 'bar':
            if graph_subtype == 'Grouped Bar':
                if not z_feature:
                    return jsonify({"message": "Z feature is required for Grouped Bar."}), 400
                # Group the data
                grouped_data = data_frame.groupby([x_feature, z_feature])[y_feature].sum().reset_index()
                response_data['grouped_data'] = grouped_data.to_dict(orient='records')
                response_data['title'] = f'Grouped Bar Chart of {y_feature} by {x_feature} and {z_feature}'

            elif graph_subtype == 'horizontal bar':
                response_data['x'] = data_frame[y_feature].tolist()
                response_data['y'] = data_frame[x_feature].tolist()
                response_data['title'] = f'Horizontal Bar Chart of {y_feature} by {x_feature}'

            elif graph_subtype == 'Stacked Bar':
                if not z_feature:
                    return jsonify({"message": "Z feature is required for Stacked Bar."}), 400
                # Prepare data for stacked bar chart
                stacked_data = data_frame.groupby([z_feature, x_feature])[y_feature].sum().unstack().fillna(0)
                response_data['stacked_data'] = stacked_data.to_dict(orient='index')
                response_data['title'] = f'Stacked Bar Chart of {y_feature} by {z_feature} with {x_feature}'

            else:
                # Default  Bar or other subtypes
                response_data['x'] = data_frame[x_feature].tolist()
                response_data['y'] = data_frame[y_feature].tolist()
                response_data['name'] = y_feature
                response_data['title'] = f'Bar Chart of {y_feature} by {x_feature}'
                
                # You can also add an optional bar width
                response_data['width'] = 0.5

        elif graph_type == 'scatter':
            response_data['x'] = data_frame[x_feature].tolist()
            response_data['y'] = data_frame[y_feature].tolist()
            response_data['title'] = f'Scatter Plot of {y_feature} vs {x_feature}'

        elif graph_type == 'line':
            response_data['x'] = data_frame[x_feature].tolist()
            response_data['y'] = data_frame[y_feature].tolist()
            
            if graph_subtype == 'Dashed Line':
                response_data['line_type'] = 'dash'
            elif graph_subtype == 'Smooth Line':
                response_data['line_type'] = 'smooth'
            else:
                response_data['line_type'] = 'linear'
            
            response_data['title'] = f'Line Chart of {y_feature} vs {x_feature}'


        elif graph_type == 'histogram':
            response_data['x'] = data_frame[x_feature].tolist()  # Histogram based on Y feature
            response_data['title'] = f'Histogram of {x_feature}'

        elif graph_type == 'boxplot':
            if graph_subtype == 'Grouped Box':
                if not x_feature:
                    return jsonify({"message": "X feature is required for Grouped Box."}), 400
                # Box plot grouped by Z feature
                grouped_data = data_frame.groupby(x_feature)[y_feature].apply(list).to_dict()
                response_data['grouped_data'] = grouped_data
                response_data['title'] = f'Box Plot of {y_feature} grouped by {x_feature}'
            else:
                # Single box plot
                response_data['y'] = data_frame[y_feature].tolist()
                response_data['title'] = f'Box Plot of {y_feature}'

        elif graph_type == 'errorbars':
            # Assuming error is standard deviation
            if graph_subtype == 'Error Bars X':
                error = data_frame[x_feature].std()
                response_data['x'] = data_frame[x_feature].tolist()
                response_data['y'] = data_frame[y_feature].tolist()
                response_data['error_x'] = [error] * len(data_frame)
                response_data['title'] = f'Error Bars (X) for {y_feature} vs {x_feature}'
            elif graph_subtype == 'Error Bars Y':
                error = data_frame[y_feature].std()
                response_data['x'] = data_frame[x_feature].tolist()
                response_data['y'] = data_frame[y_feature].tolist()
                response_data['error_y'] = [error] * len(data_frame)
                response_data['title'] = f'Error Bars (Y) for {y_feature} vs {x_feature}'
            else:
                return jsonify({"message": "Invalid subtype for Error Bars."}), 400

        elif graph_type == 'heatmap':
            try:
                # Check if the features exist in the DataFrame
                if x_feature not in data_frame.columns or y_feature not in data_frame.columns or z_feature not in data_frame.columns:
                    return jsonify({'message': 'One or more features do not exist in the dataset.'}), 400

                # Ensure unique values for X and Y features
                x_values = data_frame[x_feature].unique()
                y_values = data_frame[y_feature].unique()

                # Create a pivot table for the heatmap
                heatmap_data = data_frame.pivot_table(values=z_feature, index=y_feature, columns=x_feature, aggfunc=np.mean)
                heatmap_data.fillna(0, inplace=True)

                # Convert to a 2D array
                z_values = heatmap_data.values.tolist()  # 2D array for heatmap
                x_values = heatmap_data.columns.tolist()  # X-axis labels
                y_values = heatmap_data.index.tolist()     # Y-axis labels

                # Ensure that x_values and y_values are unique
                x_values = list(dict.fromkeys(x_values))
                y_values = list(dict.fromkeys(y_values))

                return jsonify({
                    'z': z_values,
                    'x': x_values,
                    'y': y_values,
                    'title': 'Heatmap of ' + z_feature
                })
            except Exception as e:
                print(f"Error processing heatmap: {str(e)}")
                return jsonify({'message': 'Error processing graph data.'}), 500
            
        elif graph_type == 'violin':
            response_data['y'] = data_frame[y_feature].tolist()
            if z_feature:
                response_data['x'] = data_frame[z_feature].tolist()
                response_data['title'] = f'Violin Plot of {y_feature} grouped by {z_feature}'
            else:
                response_data['title'] = f'Violin Plot of {y_feature}'

        else:
            return jsonify({"message": "Unsupported graph type."}), 400

        return jsonify(response_data)
    except Exception as e:
        print(f"Error processing graph: {str(e)}")
        return jsonify({"message": "Error processing graph data."}), 500
    
import os
import uuid
import base64
from flask import request, jsonify
from io import BytesIO
from PIL import Image

@app.route('/save_graph', methods=['POST'])
def save_graph():
    try:
        # Get the graph data from the request
        graph_data = request.json
        graph_image_base64 = graph_data.get('graph_image') # Base64-encoded image string

        if not graph_image_base64:
            return jsonify({"message": "No graph image data found."}), 400

        # Decode the base64 string to image data
        image_data = base64.b64decode(graph_image_base64.split(',')[1])  # Remove the "data:image/png;base64," part

        # Create the folder to save graphs if it doesn't exist
        folder_path = os.path.join(app.static_folder, 'saved_graphs')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate a unique ID for the graph and create the filename
        graph_id = str(uuid.uuid4())
        graph_filename = f"saved_graph_{graph_id}.png"
        graph_filepath = os.path.join(folder_path, graph_filename)

        # Save the image to the server
        image = Image.open(BytesIO(image_data))
        image.save(graph_filepath)

        # Save other graph metadata (e.g., type, subtype, etc.) to a file or database if necessary
        graph_metadata = {
            'x': graph_data.get('x'),
            'y': graph_data.get('y'),
            'z': graph_data.get('z'),
            'graph_type': graph_data.get('graph_type'),
            'graph_subtype': graph_data.get('graph_subtype'),
            'color': graph_data.get('color'),
            'marker_size': graph_data.get('marker_size'),
            'show_grid': graph_data.get('show_grid'),
            'graph_filename': graph_filename,
        }

        # You can optionally store the metadata (e.g., in a database or a JSON file) if necessary

        return jsonify({"message": "Graph saved successfully!", "graph_id": graph_id, "graph_filename": graph_filename}), 200
    except Exception as e:
        print(f"Error saving graph: {str(e)}")
        return jsonify({"message": "Error saving graph."}), 500





# Analysis
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import json
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import plotly





label_encoders = {}

@app.route('/analysis')
def analysis_index():
    return render_template('analysis.html')

@app.route('/get_features', methods=['GET'])
def get_features():
    features = data_frame.columns.tolist()
    return jsonify(features)

@app.route('/correlation_matrix', methods=['GET'])
def correlation_matrix():
    global data_frame
    if data_frame.empty:
        return jsonify({"message": "No data available!"})
    
    data_encoded = data_frame.copy()
    for column in data_frame.columns:
        if data_frame[column].dtype == 'object' or data_frame[column].dtype.name == 'category':
            le = LabelEncoder()
            data_encoded[column] = le.fit_transform(data_frame[column])
    
    correlation_matrix = data_encoded.corr()
    num_features = len(correlation_matrix.columns)
    fig_width = max(600, num_features * 30)
    fig_height = max(400, num_features * 25)

    fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu', title='Correlation Heatmap',
                    labels=dict(x='Features', y='Features', color='Correlation Coefficient'), text_auto=True, aspect="auto")
    fig.update_layout(width=fig_width, height=fig_height)
    heatmap_json = fig.to_json()

    return jsonify(heatmap_json)



@app.route('/descriptive_statistics', methods=['GET'])
def descriptive_statistics():
    global data_frame
    if data_frame.empty:
        return jsonify({"error": "No data uploaded"})

    # Calculate descriptive statistics
    desc_stats = data_frame.describe().T

    # Convert the descriptive statistics DataFrame to HTML with custom classes for styling
    desc_stats_html = desc_stats.to_html(classes="analysis-table", border=0)

    return jsonify({"table": desc_stats_html})


@app.route('/scatter_matrix', methods=['GET'])
def scatter_matrix():
    global data_frame
    if data_frame.empty:
        return jsonify({"error": "No data uploaded"})

    # Create a scatter matrix plot
    fig = px.scatter_matrix(data_frame,
        dimensions=data_frame.select_dtypes(include=[np.number]).columns.tolist(),  # Automatically select numeric columns
        color=data_frame.select_dtypes(include=[object]).columns[0] if not data_frame.select_dtypes(include=[object]).empty else None  # Color by first categorical column if exists
    )
    scatter_matrix_json = fig.to_json()

    return jsonify(scatter_matrix_json)




@app.route('/regression_analysis', methods=['POST'])
def regression_analysis():
    global data_frame
    if not data_frame.empty:
        X_columns = request.json.get('independent_vars')
        y_column = request.json.get('dependent_var')

        # Prepare the independent and dependent variables
        X = data_frame[X_columns]
        y = data_frame[y_column]

        # Encode any categorical variables in X and y
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':  # Detect categorical columns
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

        if y.dtype == 'object':  # Encode the dependent variable if it's categorical
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoders[y_column] = le

        # Fit the regression model
        model = LinearRegression()
        model.fit(X, y)

        # Get coefficients and intercept
        coefs = model.coef_.tolist()
        intercept = model.intercept_

        # Create sorted x values for the regression line
        x_sorted = np.sort(X[X_columns[0]].values)
        
        # For multiple independent variables, we need to create a proper input array
        if len(X_columns) > 1:
            # Use mean values for other variables
            X_means = X.mean()
            X_line = np.zeros((len(x_sorted), len(X_columns)))
            X_line[:, 0] = x_sorted
            for i in range(1, len(X_columns)):
                X_line[:, i] = X_means[i]
        else:
            X_line = x_sorted.reshape(-1, 1)

        # Calculate predicted y values for the sorted x values
        y_pred = model.predict(X_line)

        # Return both original data and sorted data for the regression line
        return jsonify({
            "coefficients": coefs,
            "intercept": intercept,
            "original_x": X[X_columns[0]].tolist(),
            "original_y": y.tolist(),
            "line_x": x_sorted.tolist(),
            "line_y": y_pred.tolist()
        })

    return jsonify({"error": "No data uploaded"})

import io
import base64
from flask import Flask, jsonify, request
import pandas as pd
import plotly.graph_objects as go


@app.route('/time_series_analysis', methods=['POST'])
def time_series_analysis():
    global data_frame
    if not data_frame.empty:
        time_series_col = request.json.get('time_series_col')
        feature = request.json.get('other_features', [])[0]  # Get the first and only feature
        if time_series_col in data_frame.columns and feature in data_frame.columns:
            # Create the trace data
            trace = {
                'x': data_frame[time_series_col].tolist(),
                'y': data_frame[feature].tolist(),
                'mode': 'lines',
                'name': feature,
                'type': 'scatter'
            }

            # Create the layout
            layout = {
                'title': f'{feature} over {time_series_col}',
                'xaxis': {
                    'title': time_series_col,
                    'tickformat': '%Y-%m-%d',
                    'dtick': "M1",
                    'tickangle': 90,
                    'showgrid': True,
                    'tickmode': 'auto'
                },
                'yaxis': {
                    'title': feature,
                    'showgrid': True
                },
                'template': 'plotly_dark'
            }

            # Return the plot data in the response
            return jsonify({
                "plots": {
                    feature: {
                        'data': [trace],
                        'layout': layout
                    }
                }
            })
    return jsonify({"error": "No data uploaded or invalid column"})




from scipy import stats
from flask import request, jsonify

@app.route('/hypothesis_testing', methods=['POST'])
def hypothesis_testing():
    global data_frame
    if not data_frame.empty:
        sample_col = request.json.get('sample_col')
        pop_mean = float(request.json.get('population_mean'))
        if sample_col in data_frame.columns:
            sample_data = data_frame[sample_col]
            t_stat, p_val = stats.ttest_1samp(sample_data.dropna(), pop_mean)
            
            # Define hypotheses
            null_hypothesis = f"The population mean is equal to {pop_mean}"
            alternative_hypothesis = f"The population mean is not equal to {pop_mean}"
            
            # Decision to reject null hypothesis (commonly using alpha = 0.05)
            alpha = 0.05
            reject_null = bool(p_val < alpha)  # Convert numpy boolean to Python boolean

            return jsonify({
                "t_statistic": t_stat,
                "p_value": p_val,
                "null_hypothesis": null_hypothesis,
                "alternative_hypothesis": alternative_hypothesis,
                "reject_null": reject_null
            })
    return jsonify({"error": "No data uploaded or invalid column"})





@app.route('/pca', methods=['POST'])
def perform_pca():
    global data_frame
    if data_frame.empty:
        return jsonify({"error": "No data uploaded"}), 400

    data_json = request.get_json()
    num_components = int(data_json.get('num_components'))
    color_column = data_json.get('color_column')

    if num_components < 1:
        return jsonify({"error": "Number of components must be at least 1"}), 400
    if color_column not in data_frame.columns:
        return jsonify({"error": f"Column '{color_column}' not found in data"}), 400

    data_encoded = data_frame.copy()
    categorical_cols = data_encoded.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if col == color_column:
            # Store encoder for the color column only, if needed
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col])
            label_encoders[col] = le
        else:
            data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded.drop(columns=[color_column]))

    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(data_scaled)

    pca_df = pd.DataFrame(pca_result, columns=[f'PCA Component {i+1}' for i in range(num_components)])
    if color_column in label_encoders:
        pca_df[color_column] = label_encoders[color_column].inverse_transform(data_encoded[color_column])
    else:
        pca_df[color_column] = data_frame[color_column].values

    fig = px.scatter_matrix(pca_df, dimensions=[f'PCA Component {i+1}' for i in range(num_components)], color=color_column)
    fig.update_traces(diagonal_visible=False)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'message': 'PCA completed', 'graph': graph_json})


import os
import uuid
import base64
import json
from flask import Flask, request, jsonify


# @app.route('/save_analysis', methods=['POST'])
# def save_analysis():
#     try:
#         # Get the analysis data from the request
#         analysis_data = request.json
#         analysis_html = analysis_data.get('analysis_html')  # HTML of the analysis (for tables)
#         analysis_image_base64 = analysis_data.get('analysis_image')  # Base64-encoded image (if any)
#         analysis_json = analysis_data.get('analysis_json')  # JSON data for non-Plotly analysis

#         # Check if we have data to save (either HTML or JSON)
#         if not analysis_html and not analysis_json and not analysis_image_base64:
#             return jsonify({"message": "No analysis data found."}), 400

#         # Save the analysis content
#         folder_path = os.path.join(app.static_folder, 'saved_analyses') 
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         # Generate unique filename for the analysis
#         analysis_id = str(uuid.uuid4())

#         # Handle HTML-based table analysis (non-Plotly)
#         html_filename = None
#         if analysis_html:
#             html_filename = f"analysis_{analysis_id}.html"
#             html_filepath = os.path.join(folder_path, html_filename)
#             with open(html_filepath, 'w') as html_file:
#                 html_file.write(analysis_html)

#         # Handle JSON-based analysis
#         json_filename = None
#         if analysis_json:
#             json_filename = f"analysis_{analysis_id}.json"
#             json_filepath = os.path.join(folder_path, json_filename)
#             with open(json_filepath, 'w') as json_file:
#                 json.dump(analysis_json, json_file)

#         # Handle Plotly image if available
#         image_filename = None
#         if analysis_image_base64:
#             image_filename = f"analysis_{analysis_id}.png"
#             image_filepath = os.path.join(folder_path, image_filename)

#             # Decode the base64 image data and save it
#             image_data = base64.b64decode(analysis_image_base64.split(',')[1])  # Removing the 'data:image/png;base64,' part
#             with open(image_filepath, 'wb') as image_file:
#                 image_file.write(image_data)

#         # Store analysis metadata (type, parameters, filenames)
#         analysis_metadata = {
#             'analysis_type': analysis_data.get('analysis_type'),
#             'analysis_subtype': analysis_data.get('analysis_subtype'),
#             'parameters': analysis_data.get('parameters'),
#             'html_filename': html_filename,
#             'json_filename': json_filename,
#             'image_filename': image_filename  # Save the image filename if available
#         }

#         # Return the response with metadata
#         return jsonify({
#             "message": "Analysis saved successfully!",
#             "analysis_id": analysis_id,
#             "html_filename": html_filename,
#             "json_filename": json_filename,
#             "image_filename": image_filename
#         }), 200

#     except Exception as e:
#         print(f"Error saving analysis: {str(e)}")
#         return jsonify({"message": "Error saving analysis."}), 500

@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    try:
        # Get the analysis data from the request
        analysis_data = request.json
        analysis_html = analysis_data.get('analysis_html')  # HTML of the analysis (for tables)
        analysis_image_base64 = analysis_data.get('analysis_image')  # Base64-encoded image (if any)
        analysis_json = analysis_data.get('analysis_json')
        analysis_type = analysis_data.get('analysis_type')   # JSON data for non-Plotly analysis
        print("Analysis Type = ",analysis_type )

        # Check if we have data to save (either HTML or JSON)
        if not analysis_html and not analysis_json and not analysis_image_base64:
            return jsonify({"message": "No analysis data found."}), 400

        # Save the analysis content
        folder_path = os.path.join(app.static_folder, 'saved_analyses') 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate unique filename for the analysis
        analysis_id = str(uuid.uuid4())

        # Handle HTML-based table analysis (non-Plotly)
        html_filename = None
        if analysis_html:
            html_filename = f"analysis_{analysis_id}.html"
            html_filepath = os.path.join(folder_path, html_filename)
            with open(html_filepath, 'w') as html_file:
                html_file.write(analysis_html)

        # Handle JSON-based analysis
        json_filename = None
        if analysis_json:
            json_filename = f"analysis_{analysis_id}.json"
            json_filepath = os.path.join(folder_path, json_filename)
            with open(json_filepath, 'w') as json_file:
                json.dump(analysis_json, json_file)

        # Handle Plotly image if available
        image_filename = None
        if analysis_image_base64:
            image_filename = f"analysis_{analysis_id}.png"
            image_filepath = os.path.join(folder_path, image_filename)

            # Decode the base64 image data and save it
            image_data = base64.b64decode(analysis_image_base64.split(',')[1])  # Removing the 'data:image/png;base64,' part
            with open(image_filepath, 'wb') as image_file:
                image_file.write(image_data)
        
        # Save the relative path for image
        image_path = None
        if image_filename:
            image_path = url_for('static', filename=f'saved_analyses/{image_filename}')

        # Store analysis metadata (type, parameters, filenames)
        analysis_metadata = {
            'analysis_type': analysis_type,
            'analysis_subtype': analysis_data.get('analysis_subtype'),
            'parameters': analysis_data.get('parameters'),
            'html_filename': html_filename,
            'json_filename': json_filename,
            'image_filename': image_filename,  # Save the image filename if available
            'image_path': image_path  # Relative path to the image for rendering
        }

        # Return the response with metadata
        return jsonify({
            "message": "Analysis saved successfully!",
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,  # Send the analysis type in the response
            "html_filename": html_filename,
            "json_filename": json_filename,
            "image_filename": image_filename
        }), 200

    except Exception as e:
        print(f"Error saving analysis: {str(e)}")
        return jsonify({"message": "Error saving analysis."}), 500




import os
import json
import base64
from flask import Flask, render_template, url_for

@app.route('/generate_report')
def generate_report():
    graphs_folder = os.path.join(app.static_folder, 'saved_graphs')
    analyses_folder = os.path.join(app.static_folder, 'saved_analyses')
    saved_images_folder = os.path.join(app.static_folder, 'saved_images')

    graphs = []
    analyses = []

    # Process graphs
    if os.path.exists(graphs_folder):
        graph_files = [f for f in os.listdir(graphs_folder) if f.endswith('.png')]
        for graph_file in graph_files:
            graph_filepath = os.path.join('saved_graphs', graph_file)
            graph_data = {
                'graph_filename': graph_file,
                'graph_path': url_for('static', filename=graph_filepath.replace('\\', '/')),
                'graph_type': 'Type of Graph',
                'graph_subtype': 'Subtype of Graph',
                'x_title': 'X-Axis Title',
                'y_title': 'Y-Axis Title'
            }
            graphs.append(graph_data)
            print(f"Graph found: {graph_filepath}")

    # Check if analyses folder exists and list its files
    if os.path.exists(analyses_folder):
        all_files = os.listdir(analyses_folder)

        
        # Process HTML files for analysis
        html_files = [f for f in all_files if f.endswith('.html')]
        for html_file in html_files:
            html_filepath = os.path.join(analyses_folder, html_file)
            print(f"Processing HTML analysis file: {html_filepath}")
            with open(html_filepath, 'r') as html_f:
                html_content = html_f.read()

            analysis_data = {
                'analysis_type': 'Descriptive Analysis',
                'analysis_subtype': 'Subtype of HTML Analysis',
                'html_filename': html_file,
                'html_content': html_content  # HTML content to be rendered in the report
            }

            analyses.append(analysis_data)

        # Process PNG files for analysis images
        png_files = [f for f in all_files if f.endswith('.png')]
        for png_file in png_files:
            png_filepath = os.path.join(analyses_folder, png_file)
            print(f"Processing PNG analysis file: {png_filepath}")
            
            analysis_data = {
                'analysis_type': 'Image Analysis',
                'analysis_subtype': 'Subtype of Image Analysis',
                'image_filename': png_file,
                'image_path': url_for('static', filename=f'saved_analyses/{png_file}')  # URL for rendering the image
            }

            analyses.append(analysis_data)

    else:
        print(f"Analyses folder does not exist: {analyses_folder}")

    # Render report template with graphs and analyses data
    return render_template('report_template.html', graphs=graphs, analyses=analyses)




from flask import send_file, send_from_directory
import os

@app.route('/download/image/<path:filename>')
def download_image(filename):
    try:
        # Check which folder the file might be in
        possible_folders = ['saved_graphs', 'saved_analyses']
        for folder in possible_folders:
            file_path = os.path.join(app.static_folder, folder, filename)
            if os.path.exists(file_path):
                return send_file(file_path, as_attachment=True)
        return "File not found", 404
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return "Error downloading file", 500

@app.route('/download/html/<path:filename>')
def download_html(filename):
    try:
        file_path = os.path.join(app.static_folder, 'saved_analyses', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        return "File not found", 404
    except Exception as e:
        print(f"Error downloading HTML: {str(e)}")
        return "Error downloading file", 500

@app.route('/download/report')
def download_report():
    try:
        # Logic for downloading complete report
        pass
    except Exception as e:
        print(f"Error downloading report: {str(e)}")
        return "Error downloading report", 500


# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas

# @app.route('/download_report')
# def download_report():
#     # Create a PDF file
#     report_filename = 'analysis_report.pdf'
#     report_filepath = os.path.join('reports', report_filename)
    
#     c = canvas.Canvas(report_filepath, pagesize=letter)
#     c.drawString(100, 750, "Analysis and Graph Report")
    
#     # Add graphs and analyses data (simplified example)
#     c.drawString(100, 730, "Graphs:")
#     for i, graph in enumerate(graphs):
#         c.drawString(100, 710 - i*20, f"Graph {i+1}: {graph.get('graph_type')}")
    
#     c.drawString(100, 680, "Analyses:")
#     for i, analysis in enumerate(analyses):
#         c.drawString(100, 660 - i*20, f"Analysis {i+1}: {analysis.get('analysis_type')}")
    
#     c.save()
    
#     # Return the file to the user
#     return send_file(report_filepath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
 # Return success message



