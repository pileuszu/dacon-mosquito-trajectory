from config import TEST_DIR, SAMPLE_SUBMISSION_PATH, OUTPUT_DIR
import pandas as pd
from model import ConstantVelocityModel

def main():
    print("Generating Test Predictions using Constant Velocity Model...")
    
    if not TEST_DIR.exists():
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    # Load data
    test_files = sorted(TEST_DIR.glob('*.csv'))
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    print(f'Total test files: {len(test_files)}')

    # Initialize model
    model = ConstantVelocityModel()
    
    # Generate predictions
    test_pred = model.predict_batch(test_files)
    
    # Create Submission
    submission = sample_submission[['id']].merge(test_pred, on='id', how='left')
    
    output_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(output_path, index=False)
    
    print(f'\nSaved submission to {output_path}')

if __name__ == "__main__":
    main()
