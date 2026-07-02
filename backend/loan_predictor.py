# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd

# Show all CSV files in project
print("CSV Files Found:")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            print(os.path.join(root, file))

print("\nDone!")

# %%
import pandas as pd

df = pd.read_csv("datasets/train.csv")

print("Dataset Shape:", df.shape)

df.head()

# %%
import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb"):
            print(os.path.join(root, file))

# %%
import os

print("Current Folder:")
print(os.getcwd())

print("\nFiles in datasets folder:")
if os.path.exists("datasets"):
    print(os.listdir("datasets"))
else:
    print("datasets folder not found")

# %%
import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            print(os.path.join(root, file))

# %%
import os

downloads = r"C:\Users\Sivaramakrishnan\Downloads"

for file in os.listdir(downloads):
    if file.endswith(".csv") or file.endswith(".zip"):
        print(file)

# %%
import zipfile
import os

zip_path = r"C:\Users\Sivaramakrishnan\Downloads\loan-prediction-problem-dataset.zip.zip"

extract_path = r"C:\Users\Sivaramakrishnan\IDBI-ProspectAssist-AI\datasets"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction Complete!")

print("\nFiles in datasets folder:")
print(os.listdir(extract_path))

# %%
import os

print(os.listdir("datasets"))

# %%
import pandas as pd

df = pd.read_csv("datasets/train_u6lujuX_CVtuZ9i.csv")

print("Dataset Shape:", df.shape)
df.head()

# %%
print(df.columns)

# %%
df.isnull().sum()

# %%
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df.isnull().sum()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

df.head()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

df.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# %%
import joblib

joblib.dump(model, "models/loan_eligibility_model.pkl")

print("Model Saved Successfully!")

# %%
df = df.drop("Loan_ID", axis=1)

# %%
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy * 100)

# %%
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

df["EMI_Income_Ratio"] = df["LoanAmount"] / df["TotalIncome"]

df["Income_Per_Dependent"] = df["TotalIncome"] / (df["Dependents"] + 1)

# %%
df = pd.get_dummies(df, drop_first=True)

# %%
# Remove ID
df = df.drop("Loan_ID", axis=1)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Convert target only
df["Loan_Status"] = df["Loan_Status"].map({"Y":1, "N":0})

# Feature Engineering
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# One Hot Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# %%
print(df.columns.tolist())

# %%
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# %%
import pandas as pd

df = pd.read_csv("datasets/train_u6lujuX_CVtuZ9i.csv")

print(df.columns)

# %%
print(df.columns.tolist())

# %%
import pandas as pd

# Reload dataset
df = pd.read_csv("datasets/train_u6lujuX_CVtuZ9i.csv")

# Remove ID column
df = df.drop("Loan_ID", axis=1)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Feature Engineering
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# Convert target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

print(df.shape)
df.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# %%
import joblib

joblib.dump(model, "models/loan_eligibility_model.pkl")

print("Model saved successfully!")


# %%
def financial_health_score(income, loan_amount, credit_history):
    score = 50

    if income > 5000:
        score += 20

    if credit_history == 1:
        score += 20

    if loan_amount < income * 5:
        score += 10

    return min(score, 100)

print(financial_health_score(6000, 150, 1))

# %%
