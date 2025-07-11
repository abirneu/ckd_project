from django.shortcuts import render
import pandas as pd



from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



# Create your views here.
def home(request):
    return render(request, 'home.html')

def predict(request):
    # Logic for prediction goes here
    # For now, we will just render a simple template
    return render(request, 'predict.html')

def group(request):
    return render(request, 'grp.html')  # Make sure this template exists as 'grp.html'

# Helper function to process input values for CKD prediction
def get_input_values(request):
    vals = []
    for i in range(1, 25):  # Adjust based on your number of input fields
        val = request.GET.get(f'n{i}', '').strip()
        
        # Handle specific categorical fields
        if i in [6, 7, 8, 9, 19, 20, 21, 22, 23,24]:  # Categorical fields (rbc, pc, etc.)
            if val.lower() in ['normal', 'notpresent', 'no', 'good']:
                vals.append(0)
            elif val.lower() in ['abnormal', 'present', 'yes', 'poor']:
                vals.append(1)
            else:
                vals.append(0)  # Default value
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(0.0)  # Default value for numerical fields
    return vals

# CKD Prediction View
def predict(request):  
    result1 = 'NA'
    form_values = {}

    if request.method == 'GET' and request.GET:
        try:
            # Load the dataset
            dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")

            # Define categorical replacements
            replacements = {
                'rbc': {'normal': 0, 'abnormal': 1},
                'pc': {'normal': 0, 'abnormal': 1},
                'pcc': {'notpresent': 0, 'present': 1},
                'ba': {'notpresent': 0, 'present': 1},
                'htn': {'no': 0, 'yes': 1},
                'dm': {'\tyes': 1, ' yes': 1, '\tno': 0, 'no': 0, 'yes': 1},
                'cad': {'\tno': 0, 'no': 0, 'yes': 1},
                'appet': {'good': 1, 'poor': 0},
                'pe': {'no': 0, 'yes': 1},
                'ane': {'no': 0, 'yes': 1},
                'classification': {'ckd\t': 'ckd'}
            }

            # Replace categorical values
            for col, mapping in replacements.items():
                if col in dataset.columns:
                    dataset[col] = dataset[col].replace(mapping)

            # Convert target column
            if 'classification' in dataset.columns:
                dataset['classification'] = dataset['classification'].apply(
                    lambda x: 1 if x == 'ckd' else 0)

            # Convert specific columns to numeric
            for col in ['pcv', 'wc', 'rc']:
                if col in dataset.columns:
                    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

            # Fill missing values
            features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 
                       'dm', 'cad', 'appet', 'pe', 'ane']
            for feature in features:
                if feature in dataset.columns:
                    dataset[feature].fillna(dataset[feature].median(), inplace=True)

            # Clean column names
            dataset.columns = dataset.columns.str.strip()

            # Define target and features
            target_column = 'classification'
            columns_to_drop = [col for col in ['Unnamed: 0', 'id'] if col in dataset.columns]
            X = dataset.drop(columns=columns_to_drop + [target_column])
            y = dataset[target_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Get user input
            vals = get_input_values(request)
            
            # Store form values for template
            form_values = {f'n{i}': request.GET.get(f'n{i}', '') for i in range(1, 25)}
            
            # Make prediction
            scaled_input = scaler.transform([vals])
            pred = model.predict(scaled_input)
            result1 = "CKD" if pred[0] == 1 else "Not CKD"

        except Exception as e:
            result1 = f"Error: {str(e)}"
            print(f"Prediction error: {e}")

    return render(request, 'predict.html', {
        "result2": result1,
        "form_values": form_values
    })

# CKD Report View
def report_ckd(request):
    # Get all parameters from request
    context = {
        'age': request.GET.get('n1', 'Not provided'),
        'bp': request.GET.get('n2', 'Not provided'),
        'sg': request.GET.get('n3', 'Not provided'),
        'al': request.GET.get('n4', 'Not provided'),
        'su': request.GET.get('n5', 'Not provided'),
        'rbc': request.GET.get('n6', 'Not provided'),
        'pc': request.GET.get('n7', 'Not provided'),
        'pcc': request.GET.get('n8', 'Not provided'),
        'ba': request.GET.get('n9', 'Not provided'),
        'bgr': request.GET.get('n10', 'Not provided'),
        'bu': request.GET.get('n11', 'Not provided'),
        'sc': request.GET.get('n12', 'Not provided'),
        'sod': request.GET.get('n13', 'Not provided'),
        'pot': request.GET.get('n14', 'Not provided'),
        'hemo': request.GET.get('n15', 'Not provided'),
        'pcv': request.GET.get('n16', 'Not provided'),
        'wc': request.GET.get('n17', 'Not provided'),
        'rc': request.GET.get('n18', 'Not provided'),
        'htn': request.GET.get('n19', 'Not provided'),
        'dm': request.GET.get('n20', 'Not provided'),
        'cad': request.GET.get('n21', 'Not provided'),
        'appet': request.GET.get('n22', 'Not provided'),
        'pe': request.GET.get('n23', 'Not provided'),
        'ane': request.GET.get('n24', 'Not provided'),
        'result': request.GET.get('result', 'NA')
    }
    return render(request, 'report_ckd.html', context)



# Helper function to process input values for prediction
def get_input_values_grp(request):
    # vals = []
    # for j in range(1, 9):  # Assuming 8 input features (1-8)
    #     input_value = request.GET.get(f'n{j}', '').strip()  # Get input value
    #     if input_value.lower() in ['yes', 'no', 'good','poor']:  # Handle categorical inputs
    #         vals.append(1.0 if input_value.lower() == 'yes' else 0.0)
    #     else:
    #         try:
    #             vals.append(float(input_value))  # Convert to float
    #         except ValueError:
    #             vals.append(0.0)  # Default to 0.0 for invalid inputs
    # return vals
    vals = []
    for key in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']:
        val = request.GET.get(key, '').strip().lower()
        if val in ['yes', 'no']:
            vals.append(1 if val == 'yes' else 0)
        elif val in ['good', 'poor']:
            vals.append(1 if val == 'good' else 0)
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(0.0)  # Default if invalid
    return vals


def predict_grp(request):
    result1_grp = 'NA'
    form_values = {}

    if request.method == 'GET' and request.GET:
        try:
            # Load dataset
            dataset_grp = pd.read_excel(r"test_ckd\datasets\grp_pred\ckd_grp_predct_dataset.xlsx")

            # Clean & replace 'Classification'
            dataset_grp['Classification'] = dataset_grp['Classification'].str.strip().replace({
                'Group 1': 1, 'Group 2': 2, 'Group 3': 3, 'Group 4': 4,
                'Panel A': 5, 'Panel B': 6, 'Panel C': 7, 'Panel D': 8
            })

            # Drop rows with invalid 'Classification'
            dataset_grp = dataset_grp.dropna(subset=['Classification'])
            dataset_grp['Classification'] = dataset_grp['Classification'].astype(int)

            # Handle categorical features
            categorical_cols = {
                'HTN': {'no': 0, 'yes': 1},
                'DM': {'no': 0, 'yes': 1},
                'pe': {'no': 0, 'yes': 1},
                'ane': {'no': 0, 'yes': 1},
                'appt': {'good': 1, 'poor': 0}
            }

            for col, mapping in categorical_cols.items():
                if col in dataset_grp.columns:
                    dataset_grp[col] = dataset_grp[col].str.strip().str.lower().replace(mapping).astype(int)


            # Ensure numeric features
            numeric_cols = ['Age', 'Upper BP', 'Lower BP']
            for col in numeric_cols:
                dataset_grp[col] = pd.to_numeric(dataset_grp[col], errors='coerce').fillna(0)

            # Define features (X) and target (y)
            X = dataset_grp.drop(columns=['Id', 'Classification'])
            y = dataset_grp['Classification']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            # Train model
            model = LogisticRegression(
                multi_class='ovr',
                class_weight='balanced',
                max_iter=1000
            )
            model.fit(X_train, y_train)

            # Get user input
            vals = get_input_values_grp(request)
            
            # Prepare input for prediction
            input_df = pd.DataFrame([vals], columns=X_train.columns)
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            # Predict
            pred = model.predict(input_df)[0]

            # Map prediction to group/panel
            group_map = {
                1: 'Group 1', 2: 'Group 2', 3: 'Group 3', 4: 'Group 4',
                5: 'Panel A', 6: 'Panel B', 7: 'Panel C', 8: 'Panel D'
            }
            result1_grp = f"Predicted: {group_map.get(pred, 'Unknown')}"

            # Store form values for template
            form_values = {f'n{i}': request.GET.get(f'n{i}', '') for i in range(1, 9)}

        except Exception as e:
            result1_grp = f"Error: {str(e)}"
            print("Error:", e)

    return render(request, 'grp.html', {
        "result2_grp": result1_grp,
        "form_values": form_values
    })

def print_grp(request):
    # Get all parameters from the request
    context = {
        'n1': request.GET.get('n1', 'Not provided'),
        'n2': request.GET.get('n2', 'Not provided'),
        'n3': request.GET.get('n3', 'Not provided'),
        'n4': request.GET.get('n4', 'Not provided'),
        'n5': request.GET.get('n5', 'Not provided'),
        'n6': request.GET.get('n6', 'Not provided'),
        'n7': request.GET.get('n7', 'Not provided'),
        'n8': request.GET.get('n8', 'Not provided'),
        'result': request.GET.get('result', 'NA')
    }
    
    return render(request, 'report_grp.html', context)