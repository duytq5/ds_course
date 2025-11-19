import csv
import statistics
from collections import defaultdict, Counter
import re

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    PLOTTING_AVAILABLE = True
    print("Visualization libraries loaded successfully!")
except ImportError as e:
    print(f"Warning: Plotting libraries not available. Skipping plots. Error: {e}")
    PLOTTING_AVAILABLE = False


def load_csv(filename):
    """Load CSV file and return list of dictionaries"""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(filename, data, fieldnames):
    """Save data to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def extract_title(name):
    """Extract title from name (Mr., Mrs., Miss., Master., etc.)"""
    match = re.search(r'([A-Za-z]+)\.', name)
    if match:
        title = match.group(1)
        # Group rare titles
        if title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title in ['Mme']:
            return 'Mrs'
        elif title in ['Capt', 'Col', 'Major', 'Dr', 'Rev', 'Don', 'Sir', 'Jonkheer', 'Countess', 'Lady', 'Dona']:
            return 'Rare'
        return title
    return 'Unknown'


def print_missing_summary(data, label="Dataset"):
    """Print summary of missing values"""
    print(f"\n{'='*60}")
    print(f"{label} - Missing Values Summary")
    print(f"{'='*60}")

    total = len(data)
    missing = defaultdict(int)

    for row in data:
        for key, value in row.items():
            if value == '' or value is None:
                missing[key] += 1

    print(f"Total records: {total}")
    print("\nMissing values:")
    for col in sorted(missing.keys()):
        count = missing[col]
        pct = (count/total)*100
        print(f"  {col:15s}: {count:4d} ({pct:5.1f}%)")


def visualize_survival_rates(data, title="Survival Rates by Feature", filename=None, output_dir='visualizations'):
    """Create visualizations of survival rates"""
    if not PLOTTING_AVAILABLE:
        return

    # Filter out rows with missing Survived (test set)
    data_with_survival = [row for row in data if row.get('Survived', '') != '']

    if not data_with_survival:
        print(f"Skipping survival visualization - no survival data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # 1. Survival by Sex
    sex_survival = defaultdict(lambda: {'survived': 0, 'died': 0})
    for row in data_with_survival:
        sex = row['Sex']
        survived = int(row['Survived'])
        if survived == 1:
            sex_survival[sex]['survived'] += 1
        else:
            sex_survival[sex]['died'] += 1

    sexes = list(sex_survival.keys())
    survived = [sex_survival[s]['survived'] for s in sexes]
    died = [sex_survival[s]['died'] for s in sexes]

    x = range(len(sexes))
    axes[0, 0].bar(x, survived, label='Survived', alpha=0.8)
    axes[0, 0].bar(x, died, bottom=survived, label='Died', alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sexes)
    axes[0, 0].set_title('Survival by Sex')
    axes[0, 0].legend()

    # 2. Survival by Class
    class_survival = defaultdict(lambda: {'survived': 0, 'died': 0})
    for row in data_with_survival:
        pclass = row['Pclass']
        survived = int(row['Survived'])
        if survived == 1:
            class_survival[pclass]['survived'] += 1
        else:
            class_survival[pclass]['died'] += 1

    classes = sorted(class_survival.keys())
    survived = [class_survival[c]['survived'] for c in classes]
    died = [class_survival[c]['died'] for c in classes]

    x = range(len(classes))
    axes[0, 1].bar(x, survived, label='Survived', alpha=0.8)
    axes[0, 1].bar(x, died, bottom=survived, label='Died', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Class {c}' for c in classes])
    axes[0, 1].set_title('Survival by Passenger Class')
    axes[0, 1].legend()

    # 3. Survival by Embarked
    embarked_survival = defaultdict(lambda: {'survived': 0, 'died': 0})
    for row in data_with_survival:
        embarked = row.get('Embarked', 'Unknown')
        if embarked == '':
            embarked = 'Unknown'
        survived = int(row['Survived'])
        if survived == 1:
            embarked_survival[embarked]['survived'] += 1
        else:
            embarked_survival[embarked]['died'] += 1

    ports = sorted(embarked_survival.keys())
    survived = [embarked_survival[p]['survived'] for p in ports]
    died = [embarked_survival[p]['died'] for p in ports]

    x = range(len(ports))
    axes[0, 2].bar(x, survived, label='Survived', alpha=0.8)
    axes[0, 2].bar(x, died, bottom=survived, label='Died', alpha=0.8)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(ports)
    axes[0, 2].set_title('Survival by Embarkation Port')
    axes[0, 2].legend()

    # 4. Age distribution
    ages_survived = []
    ages_died = []
    for row in data_with_survival:
        age = row.get('Age', '')
        if age != '':
            age_val = float(age)
            if int(row['Survived']) == 1:
                ages_survived.append(age_val)
            else:
                ages_died.append(age_val)

    axes[1, 0].hist([ages_died, ages_survived], bins=20, label=['Died', 'Survived'], alpha=0.7)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Age Distribution by Survival')
    axes[1, 0].legend()

    # 5. Fare distribution
    fares_survived = []
    fares_died = []
    for row in data_with_survival:
        fare = row.get('Fare', '')
        if fare != '':
            fare_val = float(fare)
            if fare_val < 300:  # Remove extreme outliers for better visualization
                if int(row['Survived']) == 1:
                    fares_survived.append(fare_val)
                else:
                    fares_died.append(fare_val)

    axes[1, 1].hist([fares_died, fares_survived], bins=30, label=['Died', 'Survived'], alpha=0.7)
    axes[1, 1].set_xlabel('Fare')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Fare Distribution by Survival (capped at 300)')
    axes[1, 1].legend()

    # 6. Family size vs survival
    family_survival = defaultdict(lambda: {'survived': 0, 'died': 0})
    for row in data_with_survival:
        family_size = int(row.get('SibSp', 0)) + int(row.get('Parch', 0))
        survived = int(row['Survived'])
        if survived == 1:
            family_survival[family_size]['survived'] += 1
        else:
            family_survival[family_size]['died'] += 1

    family_sizes = sorted(family_survival.keys())
    survived = [family_survival[f]['survived'] for f in family_sizes]
    died = [family_survival[f]['died'] for f in family_sizes]

    x = range(len(family_sizes))
    axes[1, 2].bar(x, survived, label='Survived', alpha=0.8)
    axes[1, 2].bar(x, died, bottom=survived, label='Died', alpha=0.8)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(family_sizes)
    axes[1, 2].set_xlabel('Family Size (SibSp + Parch)')
    axes[1, 2].set_title('Survival by Family Size')
    axes[1, 2].legend()

    plt.tight_layout()
    if filename is None:
        filename = title.lower().replace(' ', '_') + '.png'
    else:
        filename = f'{output_dir}/{filename}'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved visualization: {filename}")
    plt.close()


def visualize_age_imputation(original_ages, imputed_ages_method1, imputed_ages_method2, output_dir='visualizations'):
    """Visualize the effect of age imputation"""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Age Imputation Comparison', fontsize=16)

    # Original (only non-missing)
    axes[0].hist(original_ages, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Original Age Distribution\n(n={len(original_ages)} non-missing)')
    axes[0].axvline(statistics.mean(original_ages), color='red', linestyle='--', label=f'Mean: {statistics.mean(original_ages):.1f}')
    axes[0].legend()

    # Method 1: Simple median
    axes[1].hist(imputed_ages_method1, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Method 1: Median Imputation\n(n={len(imputed_ages_method1)})')
    axes[1].axvline(statistics.mean(imputed_ages_method1), color='red', linestyle='--', label=f'Mean: {statistics.mean(imputed_ages_method1):.1f}')
    axes[1].legend()

    # Method 2: Group-based median
    axes[2].hist(imputed_ages_method2, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Method 2: Group-Based Median\n(n={len(imputed_ages_method2)})')
    axes[2].axvline(statistics.mean(imputed_ages_method2), color='red', linestyle='--', label=f'Mean: {statistics.mean(imputed_ages_method2):.1f}')
    axes[2].legend()

    plt.tight_layout()
    filename = f'{output_dir}/02_age_imputation_comparison.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved visualization: {filename}")
    plt.close()


def visualize_train_vs_test_distributions(train_data, test_data, output_dir='visualizations'):
    """Visualize feature distributions: train vs test comparison"""
    if not PLOTTING_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Train vs Test Feature Distributions', fontsize=16, y=0.995)

    # 1. Age Distribution
    train_ages = [float(row['Age']) for row in train_data if row.get('Age', '') != '']
    test_ages = [float(row['Age']) for row in test_data if row.get('Age', '') != '']

    axes[0, 0].hist([train_ages, test_ages], bins=20, label=['Train', 'Test'], alpha=0.7)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Age Distribution\nTrain: n={len(train_ages)}, Test: n={len(test_ages)}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Fare Distribution
    train_fares = [float(row['Fare']) for row in train_data if row.get('Fare', '') != '' and float(row['Fare']) < 300]
    test_fares = [float(row['Fare']) for row in test_data if row.get('Fare', '') != '' and float(row['Fare']) < 300]

    axes[0, 1].hist([train_fares, test_fares], bins=30, label=['Train', 'Test'], alpha=0.7)
    axes[0, 1].set_xlabel('Fare (capped at 300)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Fare Distribution\nTrain: n={len(train_fares)}, Test: n={len(test_fares)}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Passenger Class Distribution
    train_class = Counter(row['Pclass'] for row in train_data)
    test_class = Counter(row['Pclass'] for row in test_data)

    classes = sorted(set(train_class.keys()) | set(test_class.keys()))
    train_counts = [train_class.get(c, 0) for c in classes]
    test_counts = [test_class.get(c, 0) for c in classes]

    x = range(len(classes))
    width = 0.35
    axes[0, 2].bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    axes[0, 2].bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)
    axes[0, 2].set_xlabel('Passenger Class')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Passenger Class Distribution')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([f'Class {c}' for c in classes])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 4. Sex Distribution
    train_sex = Counter(row['Sex'] for row in train_data)
    test_sex = Counter(row['Sex'] for row in test_data)

    sexes = sorted(set(train_sex.keys()) | set(test_sex.keys()))
    train_counts = [train_sex.get(s, 0) for s in sexes]
    test_counts = [test_sex.get(s, 0) for s in sexes]

    x = range(len(sexes))
    axes[1, 0].bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    axes[1, 0].bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)
    axes[1, 0].set_xlabel('Sex')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Sex Distribution')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([s.capitalize() for s in sexes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. Embarked Distribution
    train_embarked = Counter(row['Embarked'] for row in train_data if row.get('Embarked', '') != '')
    test_embarked = Counter(row['Embarked'] for row in test_data if row.get('Embarked', '') != '')

    ports = sorted(set(train_embarked.keys()) | set(test_embarked.keys()))
    train_counts = [train_embarked.get(p, 0) for p in ports]
    test_counts = [test_embarked.get(p, 0) for p in ports]

    x = range(len(ports))
    axes[1, 1].bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    axes[1, 1].bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)
    axes[1, 1].set_xlabel('Embarked Port')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Embarkation Port Distribution')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(ports)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Family Size Distribution
    train_family = Counter(int(row.get('SibSp', 0)) + int(row.get('Parch', 0)) for row in train_data)
    test_family = Counter(int(row.get('SibSp', 0)) + int(row.get('Parch', 0)) for row in test_data)

    family_sizes = sorted(set(train_family.keys()) | set(test_family.keys()))
    train_counts = [train_family.get(f, 0) for f in family_sizes]
    test_counts = [test_family.get(f, 0) for f in family_sizes]

    x = range(len(family_sizes))
    axes[1, 2].bar([i - width/2 for i in x], train_counts, width, label='Train', alpha=0.8)
    axes[1, 2].bar([i + width/2 for i in x], test_counts, width, label='Test', alpha=0.8)
    axes[1, 2].set_xlabel('Family Size (SibSp + Parch)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Family Size Distribution')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(family_sizes)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'{output_dir}/04_train_vs_test_distributions.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved visualization: {filename}")
    plt.close()


def fit_imputer(train_data, method='method1'):
    """
    Fit imputation parameters on TRAINING data only (no data leakage)
    Returns a dictionary of imputation parameters
    """
    imputer_params = {'method': method}

    # Add Title feature to calculate statistics
    train_with_title = [row.copy() for row in train_data]
    for row in train_with_title:
        row['Title'] = extract_title(row['Name'])

    # === AGE IMPUTATION PARAMETERS ===
    if method == 'method1':
        # Method 1: Simple median
        ages = [float(row['Age']) for row in train_with_title if row['Age'] != '']
        imputer_params['age_median'] = statistics.median(ages)
        print(f"\n[FIT] Age median from training data: {imputer_params['age_median']:.1f}")

    else:  # method2
        # Method 2: Group-based median (by Pclass and Title)
        group_ages = defaultdict(list)

        for row in train_with_title:
            if row['Age'] != '':
                key = (row['Pclass'], row['Title'])
                group_ages[key].append(float(row['Age']))

        # Calculate median for each group
        group_medians = {}
        for key, ages in group_ages.items():
            group_medians[key] = statistics.median(ages)

        imputer_params['age_group_medians'] = group_medians

        # Overall median as fallback
        all_ages = [float(row['Age']) for row in train_with_title if row['Age'] != '']
        imputer_params['age_overall_median'] = statistics.median(all_ages)

        print(f"\n[FIT] Age group medians calculated from training data")
        print(f"  Number of groups: {len(group_medians)}")
        print(f"  Overall median (fallback): {imputer_params['age_overall_median']:.1f}")

    # === EMBARKED IMPUTATION PARAMETERS ===
    embarked_values = [row['Embarked'] for row in train_with_title if row['Embarked'] != '']
    if embarked_values:
        imputer_params['embarked_mode'] = Counter(embarked_values).most_common(1)[0][0]
        print(f"\n[FIT] Embarked mode from training data: '{imputer_params['embarked_mode']}'")

    # === FARE IMPUTATION PARAMETERS ===
    pclass_fares = defaultdict(list)
    for row in train_with_title:
        if row.get('Fare', '') != '':
            pclass_fares[row['Pclass']].append(float(row['Fare']))

    pclass_medians = {k: statistics.median(v) for k, v in pclass_fares.items()}
    imputer_params['fare_pclass_medians'] = pclass_medians
    imputer_params['fare_overall_median'] = statistics.median([f for fares in pclass_fares.values() for f in fares])

    print(f"\n[FIT] Fare medians by Pclass from training data:")
    for pclass, median in sorted(pclass_medians.items()):
        print(f"  Class {pclass}: {median:.2f}")

    return imputer_params


def transform_with_imputer(data, imputer_params, dataset_name="Dataset"):
    """
    Apply imputation parameters to data (train or test)
    Uses parameters fitted on training data to avoid data leakage
    """
    data_copy = [row.copy() for row in data]
    method = imputer_params['method']

    # Add Title feature
    for row in data_copy:
        row['Title'] = extract_title(row['Name'])

    # === AGE IMPUTATION ===
    if method == 'method1':
        median_age = imputer_params['age_median']
        missing_count = 0

        for row in data_copy:
            if row['Age'] == '':
                row['Age'] = str(median_age)
                missing_count += 1

        print(f"\n[TRANSFORM {dataset_name}] Filled {missing_count} missing ages with median: {median_age:.1f}")

    else:  # method2
        group_medians = imputer_params['age_group_medians']
        overall_median = imputer_params['age_overall_median']
        missing_count = 0

        for row in data_copy:
            if row['Age'] == '':
                key = (row['Pclass'], row['Title'])
                row['Age'] = str(group_medians.get(key, overall_median))
                missing_count += 1

        print(f"\n[TRANSFORM {dataset_name}] Filled {missing_count} missing ages using group-based medians")

    # === CABIN IMPUTATION ===
    for row in data_copy:
        row['HasCabin'] = '1' if row['Cabin'] != '' else '0'

    cabin_count = sum(1 for row in data if row['Cabin'] != '')
    print(f"\n[TRANSFORM {dataset_name}] Created 'HasCabin' binary feature")
    print(f"  Passengers with cabin: {cabin_count} ({cabin_count/len(data)*100:.1f}%)")

    # === EMBARKED IMPUTATION ===
    if 'embarked_mode' in imputer_params:
        mode_embarked = imputer_params['embarked_mode']
        missing_count = sum(1 for row in data_copy if row['Embarked'] == '')

        for row in data_copy:
            if row['Embarked'] == '':
                row['Embarked'] = mode_embarked

        if missing_count > 0:
            print(f"\n[TRANSFORM {dataset_name}] Filled {missing_count} missing Embarked with mode: '{mode_embarked}'")

    # === FARE IMPUTATION ===
    pclass_medians = imputer_params['fare_pclass_medians']
    overall_median = imputer_params['fare_overall_median']
    missing_count = sum(1 for row in data_copy if row.get('Fare', '') == '')

    if missing_count > 0:
        for row in data_copy:
            if row.get('Fare', '') == '':
                row['Fare'] = str(pclass_medians.get(row['Pclass'], overall_median))

        print(f"\n[TRANSFORM {dataset_name}] Filled {missing_count} missing Fare with Pclass-based median")

    return data_copy


def print_statistics(data, label="Dataset"):
    """Print statistical summary"""
    print(f"\n{'='*60}")
    print(f"{label} - Statistical Summary")
    print(f"{'='*60}")

    # Age statistics
    ages = [float(row['Age']) for row in data if row.get('Age', '') != '']
    if ages:
        print(f"\nAge:")
        print(f"  Mean:   {statistics.mean(ages):6.2f}")
        print(f"  Median: {statistics.median(ages):6.2f}")
        print(f"  Min:    {min(ages):6.2f}")
        print(f"  Max:    {max(ages):6.2f}")

    # Class distribution
    class_counts = Counter(row['Pclass'] for row in data)
    print(f"\nPassenger Class Distribution:")
    for pclass in sorted(class_counts.keys()):
        print(f"  Class {pclass}: {class_counts[pclass]:4d} ({class_counts[pclass]/len(data)*100:5.1f}%)")

    # Sex distribution
    sex_counts = Counter(row['Sex'] for row in data)
    print(f"\nSex Distribution:")
    for sex in sorted(sex_counts.keys()):
        print(f"  {sex:8s}: {sex_counts[sex]:4d} ({sex_counts[sex]/len(data)*100:5.1f}%)")

    # Survival rate (if available)
    if data and data[0].get('Survived', '') != '':
        survived_count = sum(1 for row in data if row['Survived'] == '1')
        print(f"\nSurvival Rate: {survived_count}/{len(data)} ({survived_count/len(data)*100:.1f}%)")


def main():
    import os

    print("="*60)
    print("TITANIC DATA VISUALIZATION AND IMPUTATION")
    print("="*60)

    # Create output directory for visualizations
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}/")

    # Load data
    print("\nLoading data...")
    train_data = load_csv('data/train.csv')
    test_data = load_csv('data/test.csv')

    # === PART 1: ORIGINAL DATA VISUALIZATION ===
    print("\n" + "="*60)
    print("PART 1: ORIGINAL DATA ANALYSIS")
    print("="*60)

    print_missing_summary(train_data, "Train Set")
    print_statistics(train_data, "Train Set (Original)")

    print_missing_summary(test_data, "Test Set")
    print_statistics(test_data, "Test Set (Original)")

    # Visualize original data
    print("\nGenerating visualizations for original data...")
    visualize_survival_rates(train_data, title="Original Data - Survival Rates by Feature",
                            filename='01_survival_rates_original.png', output_dir=output_dir)
    visualize_train_vs_test_distributions(train_data, test_data, output_dir=output_dir)

    # === PART 2: IMPUTATION ===
    print("\n" + "="*60)
    print("PART 2: DATA IMPUTATION (NO DATA LEAKAGE)")
    print("="*60)
    print("\nIMPORTANT: Fitting imputation parameters on TRAINING data only,")
    print("then applying the same parameters to both train and test sets.")
    print("This prevents data leakage and ensures realistic modeling.")

    # Original ages for comparison
    original_ages_train = [float(row['Age']) for row in train_data if row['Age'] != '']

    # Method 1: Simple median
    print("\n" + "="*60)
    print("Method 1: Simple Median Imputation")
    print("="*60)

    # Step 1: FIT on training data
    imputer_m1 = fit_imputer(train_data, method='method1')

    # Step 2: TRANSFORM both train and test using the same parameters
    train_imputed_m1 = transform_with_imputer(train_data, imputer_m1, dataset_name="Train")
    test_imputed_m1 = transform_with_imputer(test_data, imputer_m1, dataset_name="Test")

    ages_m1 = [float(row['Age']) for row in train_imputed_m1]

    # Method 2: Group-based median
    print("\n" + "="*60)
    print("Method 2: Group-Based Median Imputation")
    print("="*60)

    # Step 1: FIT on training data
    imputer_m2 = fit_imputer(train_data, method='method2')

    # Step 2: TRANSFORM both train and test using the same parameters
    train_imputed_m2 = transform_with_imputer(train_data, imputer_m2, dataset_name="Train")
    test_imputed_m2 = transform_with_imputer(test_data, imputer_m2, dataset_name="Test")

    ages_m2 = [float(row['Age']) for row in train_imputed_m2]

    # Visualize age imputation comparison
    print("\nGenerating age imputation comparison...")
    visualize_age_imputation(original_ages_train, ages_m1, ages_m2, output_dir=output_dir)

    # === PART 3: IMPUTED DATA ANALYSIS ===
    print("\n" + "="*60)
    print("PART 3: IMPUTED DATA ANALYSIS")
    print("="*60)

    print_missing_summary(train_imputed_m2, "Train Set (After Imputation - Method 2)")
    print_statistics(train_imputed_m2, "Train Set (After Imputation - Method 2)")

    # Visualize imputed data
    print("\nGenerating visualizations for imputed data...")
    visualize_survival_rates(train_imputed_m2, title="After Imputation (Method 2) - Survival Rates",
                            filename='03_survival_rates_after_imputation.png', output_dir=output_dir)

    # === PART 4: SAVE PROCESSED DATA ===
    print("\n" + "="*60)
    print("PART 4: SAVING PROCESSED DATA")
    print("="*60)

    # Save Method 1 results
    fieldnames_train_m1 = list(train_imputed_m1[0].keys())
    fieldnames_test_m1 = list(test_imputed_m1[0].keys())
    save_csv('data/train_imputed_method1.csv', train_imputed_m1, fieldnames_train_m1)
    save_csv('data/test_imputed_method1.csv', test_imputed_m1, fieldnames_test_m1)
    print("\nSaved Method 1 (Simple Median):")
    print("  - data/train_imputed_method1.csv")
    print("  - data/test_imputed_method1.csv")

    # Save Method 2 results
    fieldnames_train_m2 = list(train_imputed_m2[0].keys())
    fieldnames_test_m2 = list(test_imputed_m2[0].keys())
    save_csv('data/train_imputed_method2.csv', train_imputed_m2, fieldnames_train_m2)
    save_csv('data/test_imputed_method2.csv', test_imputed_m2, fieldnames_test_m2)
    print("\nSaved Method 2 (Group-Based Median):")
    print("  - data/train_imputed_method2.csv")
    print("  - data/test_imputed_method2.csv")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n*** IMPUTATION APPROACH (NO DATA LEAKAGE) ***")
    print("  ✓ Step 1: FIT on training data (calculate statistics)")
    print("  ✓ Step 2: TRANSFORM both train & test with SAME statistics")
    print("  ✓ Test data remains 'unseen' - no information leakage")
    print("  ✓ Realistic modeling scenario for production deployment")

    print("\nImputation Methods Applied:")
    print("  1. Age Method 1: Simple median from TRAIN data")
    print("  2. Age Method 2: Group-based median (Pclass + Title) from TRAIN data")
    print("  3. Cabin: Created 'HasCabin' binary feature (1=has cabin, 0=no cabin)")
    print("  4. Embarked: Mode from TRAIN data")
    print("  5. Fare: Pclass-based median from TRAIN data")

    print("\nKey Findings:")
    print("  - Women had much higher survival rates than men")
    print("  - First class passengers had higher survival rates")
    print("  - Children had higher survival rates")
    print("  - Having a cabin record correlates with survival (higher class)")
    print("  - Method 2 (group-based) preserves age patterns better than Method 1")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    if PLOTTING_AVAILABLE:
        print(f"\nGenerated visualizations in '{output_dir}/' directory:")
        print("  01_survival_rates_original.png          - Train data survival analysis")
        print("  02_age_imputation_comparison.png         - Comparison of imputation methods")
        print("  03_survival_rates_after_imputation.png   - Post-imputation survival analysis")
        print("  04_train_vs_test_distributions.png       - Train/Test feature distributions")
    else:
        print("\nNote: Install matplotlib to generate visualizations")

    print("\nProcessed CSV files saved in 'data/' directory:")
    print("  - train_imputed_method1.csv / test_imputed_method1.csv")
    print("  - train_imputed_method2.csv / test_imputed_method2.csv")


if __name__ == '__main__':
    main()
