import pandas as pd
import numpy as np
import os


def generate_dummy_data(filename='dummy_student_data.csv', num_records=1000):
    np.random.seed(42)

    # Generate random data
    data = {
        'G1': np.random.randint(0, 20, num_records),
        'G2': np.random.randint(0, 20, num_records),
        'G3': np.random.randint(0, 20, num_records),
        'absences': np.random.randint(0, 30, num_records),
        # 0 or 1 for binary classification
        'target': np.random.randint(0, 2, num_records)
    }

    df = pd.DataFrame(data)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create 'dataset' directory if it doesn't exist
    dataset_dir = os.path.join(script_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Save the file in the 'dataset' directory
    filepath = os.path.join(dataset_dir, filename)
    df.to_csv(filepath, index=False)
    print(f'Dummy dataset created at {filepath}')


# if __name__ == '__main__':
#     generate_dummy_data()
