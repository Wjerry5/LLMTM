import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, recall_score, accuracy_score, f1_score
import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
import json # Added for saving evaluation results

def train_xgboost_dispatcher(csv_file_path, save_model=True):
    """
    Performs complete hyperparameter tuning for the XGBoost model using GridSearchCV.
    
    Parameters:
    csv_file_path: Path to the CSV file
    save_model: Whether to save the model, default is True
    
    Returns:
    model_info: Dictionary containing the model and feature list
    """
    # --- 1. Data Preparation ---
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found '{csv_file_path}'.")
        return None

    print("--- 1. Data Preparation ---")
    features = [
        "num_edges", 
        "cyclomatic_complexity",
        "edge_locality_score",
        "ratio_nodes_ge_3",
        "ratio_nodes_eq_2"
    ]
    target = 'flag'
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Data has been split into training set ({len(X_train)} samples) and test set ({len(X_test)} samples).\n")

    # --- 2. Hyperparameter Tuning (Grid Search) ---
    print("--- 2. Hyperparameter Tuning (Grid Search) ---")

    # a. Define parameter grid
    # param_grid = {
    #     'max_depth': [4, 5, 6, 7, 8],        # Max depth of the tree
    #     'n_estimators': [100, 150, 200, 250], # Number of trees
    #     'learning_rate': [0.01, 0.05, 0.1],   # Learning rate (step size)
    #     'subsample': [0.7, 0.8, 0.9, 1.0],    # Random sampling ratio of training samples
    #     'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Random sampling ratio of training features
    #     'min_child_weight': [1, 3, 5],        # Minimum child weight
    #     'gamma': [0, 0.1, 0.2, 0.5, 1.0],           # Minimum loss reduction required for split
    #     'reg_alpha': [0, 0.01, 0.1, 1],             # L1 regularization
    #     'reg_lambda': [0.5, 1, 1.5, 2, 5]           # L2 regularization
    # }
    param_grid = {
        'max_depth': [4, 5, 6, 7, 8],        # Max depth of the tree
        'n_estimators': [100, 150, 200, 250], # Number of trees
        'learning_rate': [0.01, 0.05, 0.1],   # Learning rate (step size)
        'subsample': [0.7, 0.8, 0.9, 1.0],    # Random sampling ratio of training samples
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Random sampling ratio of training features
        'min_child_weight': [1, 3],        # Minimum child weight
        'gamma': [0, 0.1, 0.2],               # Minimum loss reduction required for split
        'reg_alpha': [0, 0.01],             # L1 regularization
        'reg_lambda': [1, 1.5]              # L2 regularization
    }
    
    # b. Use accuracy as the scoring metric
    # scoring = make_scorer(recall_score, pos_label=1)
    scoring = 'accuracy'

    # c. Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (This makes the model prioritize identifying 'hard cases')")

    # d. Initialize XGBoost classifier
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    
    # e. Set up GridSearchCV, reverting to 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),  # Revert to standard 5-fold cross-validation
        verbose=2,  # Increase verbosity
        n_jobs=-1
    )
    
    print("Grid search started, using 5-fold cross-validation...")
    grid_search.fit(X_train, y_train)
    
    print("\n--- Tuning Complete! ---")
    print(f"Best parameter combination found: {grid_search.best_params_}")
    print(f"Best average score in cross-validation: {grid_search.best_score_:.2%}\n")
    
    # Output results for all parameter combinations
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')
    print("\nTop 5 best parameter combinations and their scores:")
    for i in range(min(5, len(cv_results))):
        print(f"\nRank {i+1}:")
        print(f"Parameters: {cv_results.iloc[i]['params']}")
        print(f"Score: {cv_results.iloc[i]['mean_test_score']:.4f} (+/- {cv_results.iloc[i]['std_test_score']*2:.4f})")
    
    best_model = grid_search.best_estimator_
    
    # --- 3. XGBoost Feature Importance Analysis ---
    print("\n--- 3. XGBoost Model Feature Importance ---")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_csv_path = os.path.join(base_path ,"xgboost__feature_importance.csv")
    feature_importance.to_csv(feature_importance_csv_path, index=False)
    print(f"Feature importance weights saved to: {feature_importance_csv_path}\n")
    
    print("Feature importance ranking according to XGBoost:")
    print(feature_importance)
    
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x='Importance', y='Feature', data=feature_importance)
    # plt.title('XGBoost Feature Importance')
    # fig_path = os.path.join(base_path, "xgboost_feature_importance.png")
    # plt.savefig(fig_path)
    # plt.close()
    # print("Feature importance plot saved.\n")

    # --- 4. Evaluate Model Performance ---
    print("--- 4. Evaluating final model performance on independent test and train sets ---")
    
    # Performance on the test set (most important generalization metric)
    y_pred_test = best_model.predict(X_test)
    print("\n--- A. Final Evaluation Results on Test Set ---")
    print("Confusion Matrix:")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Error(Agent)', 'Correct(LLM)']))
    
    # Performance on the training set (to check for overfitting)
    y_pred_train = best_model.predict(X_train)
    print("\n--- B. Performance on Training Set (for comparison) ---")
    print("Confusion Matrix:")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(cm_train)
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred_train, target_names=['Error(Agent)', 'Correct(LLM)']))

    # Save detailed evaluation results
    evaluation_results = {
        'test_set_results': {
            'confusion_matrix': cm_test.tolist(),
            'classification_report': classification_report(y_test, y_pred_test, target_names=['Error(Agent)', 'Correct(LLM)'], output_dict=True)
        },
        'train_set_results': {
            'confusion_matrix': cm_train.tolist(),
            'classification_report': classification_report(y_train, y_pred_train, target_names=['Error(Agent)', 'Correct(LLM)'], output_dict=True)
        },
        'best_parameters': grid_search.best_params_,
        'cross_validation_score': grid_search.best_score_,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    evaluation_path = os.path.join(base_path,"xgboost_evaluation_results.json")
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nDetailed evaluation results saved to: {evaluation_path}")
    
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
    #             yticklabels=['True 0 (Hard Case)', 'True 1 (Easy Case)'], 
    #             xticklabels=['Predicted 0 (Use Agent)', 'Predicted 1 (Use LLM)'])
    # plt.title('XGBoost Best Model Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # confusion_matrix_path = os.path.join(base_path, "xgboost_confusion_matrix.png")
    # plt.savefig(confusion_matrix_path)
    # plt.close()
    

    if save_model:
        model_path = os.path.join(base_path, "xgboost_Pangu.joblib")
        model_info = {
            'model': best_model,
            'features': features,
            'best_params': grid_search.best_params_
        }
        joblib.dump(model_info, model_path)
        print(f"\nModel saved to: {model_path}")
    #     model_json_path = os.path.join(base_path, "xgboost_Pangu.json")
    # try:
    #     best_model.save_model(model_json_path)
    #     print(f"\nModel saved to: {model_json_path}")
    # except Exception as e:
    #     print(f"Error saving model to .json: {e}")

    return {'model': best_model, 'features': features, 'best_params': grid_search.best_params_}

def predict_with_xgboost(features_dict, model_path=None, model_info=None):
    """
    Make predictions using a trained XGBoost model.
    
    Parameters:
    features_dict: Dictionary containing feature values, e.g.:
                   {'num_nodes': 10, 'edge_density': 0.5, ...}
    model_path: Path to the model file (must be provided if model_info is None)
    model_info: Dictionary containing model information (must be provided if model_path is None)
    
    Returns:
    prediction: 0 indicates using the Agent, 1 indicates using the LLM
    probability: Probability of using the LLM
    """
    if model_info is None and model_path is None:
        raise ValueError("Either model_path or model_info must be provided")
    
    if model_info is None:
        try:
            # model_info = joblib.load(model_path)
            
            model = xgb.XGBClassifier()
            model.load_model(model_path) # Load .json file
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None
    
    # model = model_info['model']
    # features = model_info['features']
    
    # Build feature array
    features = ["num_edges","cyclomatic_complexity","edge_locality_score","ratio_nodes_ge_3","ratio_nodes_eq_2"]
    X = np.array([[features_dict[f] for f in features]])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Get probability of using the LLM
    
    return prediction, probability

# --- How to Use ---
if __name__ == '__main__':
    # Train the model
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api")
    csv_filename = os.path.join(base_path,"result_dict_five.csv")
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api/xgboost8")
    model_info = train_xgboost_dispatcher(csv_filename)
    
    # # 2. Example of making predictions with the model
    # if model_info is not None:
    #     # Example features
    #     test_features = {
    #         'cyclomatic_complexity': 5,
    #         'num_nodes': 30,
    #         'edge_locality_score': 2.5,
    #         'ratio_nodes_ge_3': 0.4,
    #         'ratio_nodes_eq_2': 0.3
    #     }
        
    #     # Method 1: Predict directly using model info
    #     prediction, probability = predict_with_xgboost(test_features, model_info=model_info)
    #     print("\nPrediction Example 1 (using model info):")
    #     print(f"Prediction: {'Use LLM' if prediction == 1 else 'Use Agent'}")
    #     print(f"Probability of using LLM: {probability:.2%}")
        
    #     # Method 2: Load model from file for prediction
    #     model_path = "/home/hb/LLMDyG_Motif/logs/example/run_one_task/judge_contain_motif/api/xgboost_best_nodes_model.joblib"
    #     prediction, probability = predict_with_xgboost(test_features, model_path=model_path)
    #     print("\nPrediction Example 2 (loading model from file):")
    #     print(f"Prediction: {'Use LLM' if prediction == 1 else 'Use Agent'}")
    #     print(f"Probability of using LLM: {probability:.2%}")