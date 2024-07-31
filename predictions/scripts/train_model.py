import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preparation import load_and_prepare_data
from generate_dummy_data import generate_dummy_data


def train_and_save_model():
    # generate data
    generate_dummy_data()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the dataset CSV relative to the script directory
    dataset_dir = os.path.join(script_dir, 'dataset')
    filepath = os.path.join(dataset_dir, 'dummy_student_data.csv')

    # Check if the dataset file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f'The dataset file was not found at {filepath}')

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create 'models' directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the model in the 'models' directory
    model_path = os.path.join(models_dir, 'student_model.pkl')
    joblib.dump(model, model_path)
    print(f'Model saved at {model_path}')


# if __name__ == '__main__':
#     train_and_save_model()
