import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Part A Loading dataset and preprocessing 

## Question No 1:

### Display the names of columns
print(df.columns)

### Display the number of rows and columns (rows, columns)
print(df.shape)

### The first 10 rows
print(df.head(10))

### The last 10 rows
print(df.tail(10))


## Question No 2:

### Display basic statistics of the dataset
print(df.describe())

### identifying the missing data
print(df.dtypes)
print(df.isnull().sum()) # There in no missing data

### Identifying imbalanced data in Churn column
print(df['Churn'].value_counts(normalize=True)) # 26.5% of the customers have churned, while 73.5% have not churned.

## Question No 3:

### Handling TotalCharges => it has object dtype, so i'll Convert it to numirec values
print(df['TotalCharges'].unique()) # Before Converting
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print(df['TotalCharges'].unique()) # After Converting

### There is no need for customerID column in prediction, so i'll drop it
df.drop('customerID', axis=1, inplace=True)
print(df.columns) # customerID column is dropped

### Encode categorical features using one-hot encoding
categorical_features = df.select_dtypes(include=['object']).columns.tolist() # Excluding numerical features
categorical_features.remove('Churn') # Remove the label
df = pd.get_dummies(df, columns=categorical_features, drop_first=True) # One-hot encoding
df = df.astype({col: int for col in df.columns if df[col].dtype == 'bool'}) # Converting the True and False values to 0s and 1s
print(df.head()) 

### Encode the lebel (yes = 1, no = 0)
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])
print(df['Churn']) # Encoded

### Normalize numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

### Split the Data into Dependent and Independent Variables
x = df.drop('Churn', axis=1).copy()
y = df['Churn'].copy()
print(x)
print(y)

## Question No 4
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

### Apply SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
print(y_train_resampled.value_counts())

## Question No 5

### I'll use random forest Beacause of:
### - 1. It can handle both numerical and categorical data.
### - 2. It is robust to overfitting.
### - 3. It can handle imbalanced datasets well.
### - 4. Insensitive for scaling

### Train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train_resampled, y_train_resampled)


## Question No 6
y_pred = rf_clf.predict(x_test)
y_pred_proba = rf_clf.predict_proba(x_test)[:, 1]  # For AUC-ROC

### Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

### Accuracy will be misleading because of the imbalanced data
### when missing a churner means losing a customer, for that recall will be more important 
### It measures how many actual churners were correctly predicted

## Question No 7
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

rf= RandomForestClassifier(random_state=42)

grid_search  = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train_resampled, y_train_resampled)


best_rf_grid = grid_search.best_estimator_
print("Best Parameters Found:\n", grid_search.best_params_)

def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model("Original Random Forest", rf_clf, x_test, y_test)

evaluate_model("Tuned Random Forest", best_rf_grid, x_test, y_test)
