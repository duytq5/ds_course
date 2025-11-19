"""
Chi-Square Test for Feature Selection
Target: Survived (Titanic Dataset)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
import argparse


def load_and_prepare_data(file_path):
    """
    Load dataset and prepare features for Chi-square test
    Chi-square requires non-negative features
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if 'Survived' exists
    if 'Survived' not in df.columns:
        raise ValueError("Target column 'Survived' not found in dataset")

    return df


def preprocess_features(df):
    """
    Preprocess features for Chi-square test:
    - Handle missing values
    - Encode categorical variables
    - Ensure all values are non-negative
    """
    print("\n" + "="*60)
    print("PREPROCESSING FEATURES")
    print("="*60)

    # Separate target and features
    y = df['Survived']
    X = df.drop(['Survived'], axis=1)

    # Drop non-useful columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
    X = X.drop([col for col in columns_to_drop if col in X.columns], axis=1)

    print(f"\nFeatures to analyze: {X.columns.tolist()}")

    # Handle missing values
    print("\nMissing values before imputation:")
    print(X.isnull().sum())

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            # Fill categorical with mode
            mode_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
            X.loc[:, col] = X[col].fillna(mode_value)
        else:
            # Fill numerical with median
            X.loc[:, col] = X[col].fillna(X[col].median())

    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            print(f"\nEncoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Ensure all values are non-negative (required for Chi-square)
    print("\nFeature statistics:")
    print(X.describe())

    return X, y, X.columns.tolist()


def perform_chi_square_test(X, y, k=2):
    """
    Perform Chi-square test and return detailed scores

    Args:
        X: Feature matrix
        y: Target variable
        k: Number of best features to select

    Returns:
        DataFrame with scores and selected features
    """
    print("\n" + "="*60)
    print("CHI-SQUARE TEST RESULTS")
    print("="*60)

    # Calculate Chi-square scores
    chi2_scores, p_values = chi2(X, y)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Feature': X.columns,
        'Chi2_Score': chi2_scores,
        'P_Value': p_values,
        'Significant': p_values < 0.05
    })

    # Sort by Chi-square score (descending)
    results_df = results_df.sort_values('Chi2_Score', ascending=False)
    results_df['Rank'] = range(1, len(results_df) + 1)

    # Reorder columns
    results_df = results_df[['Rank', 'Feature', 'Chi2_Score', 'P_Value', 'Significant']]

    print(f"\nTotal features analyzed: {len(results_df)}")
    print(f"\nDetailed Chi-Square Scores:")
    print("-" * 80)
    print(results_df.to_string(index=False))

    # Select K best features
    print("\n" + "="*60)
    print(f"K-BEST FEATURES (k={k})")
    print("="*60)

    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)


    # Save results to CSV
    output_file = f'chi_square_results_k{k}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")


    return results_df, X_selected


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Perform Chi-Square test for feature selection on Titanic dataset'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='Number of best features to select (default: 2)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/train.csv',
        help='Path to dataset (default: data/train.csv)'
    )

    args = parser.parse_args()

    print("="*60)
    print("CHI-SQUARE FEATURE SELECTION")
    print("="*60)
    print(f"Parameters:")
    print(f"  - k (number of features): {args.k}")
    print(f"  - Dataset: {args.data}")
    print("="*60)

    # Load data
    df = load_and_prepare_data(args.data)

    # Preprocess
    X, y, feature_names = preprocess_features(df)

    # Perform Chi-square test
    results_df, X_selected = perform_chi_square_test(X, y, k=args.k)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Target variable: Survived")
    print(f"Total samples: {len(y)}")
    print(f"Class distribution:")
    print(f"  - Not Survived (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  - Survived (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"\nTotal features analyzed: {len(feature_names)}")
    print(f"Selected features shape: {X_selected.shape}")
    print("="*60)


if __name__ == "__main__":
    main()
