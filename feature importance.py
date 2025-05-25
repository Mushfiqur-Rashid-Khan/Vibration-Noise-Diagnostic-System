import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("/content/Vibration_Dataset.csv")

# Define sensor features and fault labels
faults = [
    'Wiring Reversal', 'Offset Bias', 'Sensor Aging (Degradation)',
    'Thermal Drift', 'Radiation-Induced Degradation',
    'High Noise Levels', 'Gas Leakage', 'Signal Drift'
]
sensor_features = df.drop(columns=faults)

for fault in faults:
    print(f"\n--- Fault: {fault} ---")

    X = sensor_features
    y = df[fault]  # Binary classification (0 or 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Gradient Boosting model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict & Report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance plot
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1]
    top_features = X.columns[sorted_idx[:10]]
    top_importances = importance[sorted_idx[:10]]

    plt.figure(figsize=(14, 5))
    ax = sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f"Top Contributing Features for {fault}")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Sensor Features")

    # Add importance score annotations next to bars
    for i, v in enumerate(top_importances):
        ax.text(v + 0.001, i, f"{v:.4f}", color='black', va='center')

    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv("/content/Vibration_Dataset.csv")

# List of fault columns
faults = [
    'Wiring Reversal', 'Offset Bias', 'Sensor Aging (Degradation)',
    'Thermal Drift', 'Radiation-Induced Degradation',
    'High Noise Levels', 'Gas Leakage', 'Signal Drift'
]

# Drop target columns to get only sensor features
sensor_features = df.drop(columns=faults)

# Iterate through each fault
for fault in faults:
    print(f"\n--- Fault: {fault} ---")

    X = sensor_features
    y = df[fault]  # Binary classification (0 or 1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the LightGBM classifier
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Get feature importances
    # Get and normalize feature importances
    raw_importance = model.feature_importances_
    importance = raw_importance / raw_importance.sum()  # Normalize the importances
    sorted_idx = importance.argsort()[::-1]
    top_features = X.columns[sorted_idx[:]]
    top_importances = importance[sorted_idx[:]]


    # Plot feature importance
    plt.figure(figsize=(20, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    for index, value in enumerate(top_importances):
        plt.text(value + 0.5, index, f'{value:.4f}', va='center')
    plt.title(f"Top Contributing Features for Fault: {fault}")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Sensor Features")
    plt.tight_layout()
    plt.show()

    # Create a DataFrame for importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance Score': importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance Score', ascending=False)

    # Display as table
    print(f"\nFeature Importance Table for Fault: {fault}")
    print(importance_df.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Load your dataset
df = pd.read_csv("/content/Vibration_Dataset.csv")

# List of fault columns
faults = [
    'Wiring Reversal', 'Offset Bias', 'Sensor Aging (Degradation)',
    'Thermal Drift', 'Radiation-Induced Degradation',
    'High Noise Levels', 'Gas Leakage', 'Signal Drift'
]

# Drop target columns to get only sensor features
sensor_features = df.drop(columns=faults)

# Iterate through each fault
for fault in faults:
    print(f"\n--- Fault: {fault} ---")

    X = sensor_features
    y = df[fault]  # Binary classification (0 or 1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the LightGBM classifier
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Get and normalize feature importances
    raw_importance = model.feature_importances_
    importance = raw_importance / raw_importance.sum()  # Normalize
    sorted_idx = importance.argsort()[::-1]
    top_features = X.columns[sorted_idx]
    top_importances = importance[sorted_idx]

    # Plot feature importance
    plt.figure(figsize=(12, len(top_features) * 0.4))
    ax = sns.barplot(x=top_importances, y=top_features, palette="viridis")

    # Annotate bars with importance scores
    for i, (value, name) in enumerate(zip(top_importances, top_features)):
        ax.text(value + 0.005, i, f"{value:.4f}", va='center', fontsize=9)

    plt.title(f"Top Contributing Features for Fault: {fault}", fontsize=14)
    plt.xlabel("Normalized Feature Importance Score", fontsize=12)
    plt.ylabel("Sensor Features", fontsize=12)
    plt.xlim(0, max(top_importances) + 0.05)  # Add right margin
    plt.tight_layout()
    plt.show()

    # Create and print table
    importance_df = pd.DataFrame({
        'Feature': top_features,
        'Importance Score': top_importances
    })

    print(f"\nFeature Importance Table for Fault: {fault}")
    print(importance_df.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Load your dataset
df = pd.read_csv("/content/Vibration_Dataset.csv")

# List of fault columns
faults = [
    'Wiring Reversal', 'Offset Bias', 'Sensor Aging (Degradation)',
    'Thermal Drift', 'Radiation-Induced Degradation',
    'High Noise Levels', 'Gas Leakage', 'Signal Drift'
]

# Drop target columns to get only sensor features
sensor_features = df.drop(columns=faults)

# Iterate through each fault
for fault in faults:
    print(f"\n--- Fault: {fault} ---")

    X = sensor_features
    y = df[fault]  # Binary classification (0 or 1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the LightGBM classifier
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Get and normalize feature importances
    raw_importance = model.feature_importances_
    importance = raw_importance / raw_importance.sum()

    # Create DataFrame and remove zero-importance features
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance Score': importance
    })
    importance_df = importance_df[importance_df['Importance Score'] > 0]
    importance_df = importance_df.sort_values(by='Importance Score', ascending=False)

    # Plot
    plt.figure(figsize=(12, len(importance_df) * 0.4))
    ax = sns.barplot(
        x=importance_df['Importance Score'],
        y=importance_df['Feature'],
        palette="viridis"
    )
    for i, (value, name) in enumerate(zip(importance_df['Importance Score'], importance_df['Feature'])):
        ax.text(value + 0.005, i, f"{value:.4f}", va='center', fontsize=9)

    plt.title(f"Top Contributing Features for {fault}", fontsize=14)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Sensor Features", fontsize=12)
    plt.xlim(0, importance_df['Importance Score'].max() + 0.05)
    plt.tight_layout()
    plt.show()

    # Print Table
    print(f"\nFeature Importance Table for {fault}")
    print(importance_df.to_string(index=False))
