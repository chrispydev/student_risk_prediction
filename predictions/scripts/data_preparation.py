import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data['total_absences'] = data['absences']

    features = ['G1', 'G2', 'G3', 'total_absences']
    X = data[features]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
