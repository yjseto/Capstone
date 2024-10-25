import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=20):
    """
    Select top k features based on ANOVA F-value
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X[selected_features], selected_features

def prepare_predictors(df, selected_features=None):
    """
    Modified prepare_predictors with feature selection and improved preprocessing
    """
    df = df.copy()
    
    exclude_features = ['Crash_Number', 'Crash_Severity', 'severity_grouped',
                       'serious_with_fatalities', 'fatalities', 'serious',
                       'minor', 'pop10', 'Number_of_Motorized_units',
                       'Airbags', 'Unnamed: 0', 'severity_score',
                       'is_no_injury', 'is_moderate', 'is_severe']

    categorical_columns = []
    numeric_columns = []

    for col in df.columns:
        if col not in exclude_features and col != 'area':
            if df[col].dtype == 'object':
                categorical_columns.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)

    X = pd.DataFrame(index=df.index)

    # Handle numeric columns with robust scaling
    if numeric_columns:
        X_numeric = df[numeric_columns].copy()
        
        # Use RobustScaler instead of StandardScaler to handle outliers
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_numeric = pd.DataFrame(scaler.fit_transform(X_numeric),
                               columns=numeric_columns,
                               index=df.index)
        X = pd.concat([X, X_numeric], axis=1)

    # Improved categorical encoding with frequency encoding
    for col in categorical_columns:
        df[col] = df[col].fillna('missing')
        freq_encoding = df[col].value_counts(normalize=True)
        X[f'{col}_freq'] = df[col].map(freq_encoding)

    # Return only selected features if provided
    if selected_features is not None:
        X = X[selected_features]

    return X

def cross_validate_model(X, y, areas, n_splits=5):
    """
    Perform k-fold cross-validation with stratification
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        areas_train = areas.iloc[train_idx]
        
        model = MixedLM(y_train,
                       sm.add_constant(X_train),
                       groups=areas_train)
        
        try:
            results = model.fit(method='lbfgs', maxiter=1000)
            y_pred = np.clip(results.predict(sm.add_constant(X_val)), 0, 1)
            
            metrics, _ = get_metrics(y_val, y_pred)
            cv_scores.append(metrics)
        except:
            continue
            
    return cv_scores

def analyze_balanced_data(df, severity_type):
    """
    Enhanced analysis with cross-validation and feature selection
    """
    print(f"\nAnalyzing {severity_type} crashes with balanced data:")
    print("=" * 50)

    df = df.copy()
    
    # Create severity categories
    severity_mapping = {
        'No Apparent Injury': 0,
        'Possible Injury': 1,
        'Suspected Minor Injury': 1,
        'Suspected Serious Injury': 2,
        'Fatal Injury (Killed)': 2
    }

    df['severity_score'] = df['Crash_Severity'].map(severity_mapping)
    df['is_no_injury'] = (df['severity_score'] == 0).astype(int)
    df['is_moderate'] = (df['severity_score'] == 1).astype(int)
    df['is_severe'] = (df['severity_score'] == 2).astype(int)

    target = f'is_{severity_type}' if severity_type != 'severe' else 'is_severe'
    
    # Balance dataset
    positive_cases = df[df[target] == 1]
    negative_cases = df[df[target] == 0]
    sample_size = min(len(positive_cases), len(negative_cases))
    
    if len(positive_cases) > sample_size:
        positive_cases = positive_cases.sample(n=sample_size, random_state=42)
    if len(negative_cases) > sample_size:
        negative_cases = negative_cases.sample(n=sample_size, random_state=42)
    
    balanced_df = pd.concat([positive_cases, negative_cases]).reset_index(drop=True)

    # Prepare features and target
    X_initial = prepare_predictors(balanced_df)
    y = balanced_df[target]
    
    # Feature selection
    X, selected_features = select_features(X_initial, y, k=min(20, X_initial.shape[1]))
    
    # Prepare areas
    areas = pd.Series(balanced_df['area'].map({'rural': 0, 'urban': 1, 'suburban': 2}),
                     index=balanced_df.index)

    # Cross-validation
    cv_scores = cross_validate_model(X, y, areas)
    
    # Final model with selected features
    X = sm.add_constant(X)
    model = MixedLM(y, X, groups=areas)
    results = model.fit(method='lbfgs', maxiter=1000)

    print("\nCross-validation results:")
    metrics_df = pd.DataFrame(cv_scores)
    print(metrics_df.mean())
    print("\nMetrics std deviation:")
    print(metrics_df.std())

    print("\nSelected Features:")
    print(selected_features)

    print("\nFixed Effects (top 10 most significant):")
    fixed_effects = pd.DataFrame({
        'Coef': results.fe_params,
        'Std.Err': results.bse,
        't': results.tvalues,
        'P>|t|': results.pvalues
    })
    print(fixed_effects.sort_values('P>|t|').head(10))

    print("\nRandom Effects:")
    print(f"Area variance: {results.cov_re.values[0][0]:.4f}")

    return results, cv_scores, selected_features

def get_metrics(y_true, y_pred_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Calculate metrics with multiple thresholds
    """
    metrics_by_threshold = {}

    for threshold in thresholds:
        y_pred = (y_pred_prob > threshold).astype(int)
        metrics_by_threshold[threshold] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_pred_prob),
            'MSE': np.mean((y_true - y_pred_prob) ** 2),
            'RMSE': np.sqrt(np.mean((y_true - y_pred_prob) ** 2))
        }

    best_threshold = max(metrics_by_threshold.keys(),
                        key=lambda x: metrics_by_threshold[x]['F1'])

    return metrics_by_threshold[best_threshold], best_threshold

# Load your data
df = pd.read_csv('new_test_data_oct_7.csv', low_memory=False)

# Run analysis for each severity type
for severity_type in ['no_injury', 'moderate', 'severe']:
    results, cv_scores, selected_features = analyze_balanced_data(df, severity_type)