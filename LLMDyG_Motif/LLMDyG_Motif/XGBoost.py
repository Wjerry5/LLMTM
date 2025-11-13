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
    使用GridSearchCV对XGBoost模型进行完整的网格搜索超参数调优。
    
    参数：
    csv_file_path: CSV文件路径
    save_model: 是否保存模型，默认为True
    
    返回：
    model_info: 包含模型和特征列表的字典
    """
    # --- 1. 数据准备 ---
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"错误：无法找到文件 '{csv_file_path}'。")
        return None

    print("--- 1. 数据准备 ---")
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
    print(f"数据已划分为训练集 ({len(X_train)}条) 和测试集 ({len(X_test)}条)。\n")

    # --- 2. 网格搜索超参数调优 ---
    print("--- 2. 网格搜索超参数调优 (Grid Search) ---")

    # a. 定义参数网格
    # param_grid = {
    #     'max_depth': [4, 5, 6, 7, 8],          # 树的最大深度
    #     'n_estimators': [100, 150, 200, 250], # 树的数量
    #     'learning_rate': [0.01, 0.05, 0.1],  # 学习率 (步长)
    #     'subsample': [0.7, 0.8, 0.9, 1.0],     # 训练样本的随机采样比例
    #     'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # 训练特征的随机采样比例
    #     'min_child_weight': [1, 3, 5],          # 最小子节点权重
    #     'gamma': [0, 0.1, 0.2, 0.5, 1.0],                   # 分裂所需的最小损失减少
    #     'reg_alpha': [0, 0.01, 0.1, 1],              # L1正则化
    #     'reg_lambda': [0.5, 1, 1.5, 2, 5]               # L2正则化
    # }
    param_grid = {
        'max_depth': [4, 5, 6, 7, 8],          # 树的最大深度
        'n_estimators': [100, 150, 200, 250], # 树的数量
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率 (步长)
        'subsample': [0.7, 0.8, 0.9, 1.0],     # 训练样本的随机采样比例
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # 训练特征的随机采样比例
        'min_child_weight': [1, 3],          # 最小子节点权重
        'gamma': [0, 0.1, 0.2],                   # 分裂所需的最小损失减少
        'reg_alpha': [0, 0.01],              # L1正则化
        'reg_lambda': [1, 1.5]               # L2正则化
    }
    
    # b. 使用accuracy作为评分标准
    # scoring = make_scorer(recall_score, pos_label=1)
    scoring = 'accuracy'

    # c. 计算scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"计算出的scale_pos_weight为: {scale_pos_weight:.2f} (这会让模型更重视识别'难题')")

    # d. 初始化XGBoost分类器
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    
    # e. 设置GridSearchCV，恢复为5折交叉验证
    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),  # 恢复为标准的5折交叉验证
        verbose=2,  # 增加详细程度
        n_jobs=-1
    )
    
    print("网格搜索开始，将使用5折交叉验证...")
    grid_search.fit(X_train, y_train)
    
    print("\n--- 调优完成！ ---")
    print(f"找到的最佳参数组合: {grid_search.best_params_}")
    print(f"在交叉验证中，最佳的平均得分: {grid_search.best_score_:.2%}\n")
    
    # 输出所有参数组合的结果
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')
    print("\n前5个最佳参数组合及其得分:")
    for i in range(min(5, len(cv_results))):
        print(f"\n第{i+1}名:")
        print(f"参数: {cv_results.iloc[i]['params']}")
        print(f"得分: {cv_results.iloc[i]['mean_test_score']:.4f} (+/- {cv_results.iloc[i]['std_test_score']*2:.4f})")
    
    best_model = grid_search.best_estimator_
    
    # --- 3. XGBoost特征重要性分析 ---
    print("\n--- 3. XGBoost模型特征重要性 ---")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_csv_path = os.path.join(base_path ,"xgboost__feature_importance.csv")
    feature_importance.to_csv(feature_importance_csv_path, index=False)
    print(f"特征重要性权重已保存到: {feature_importance_csv_path}\n")
    
    print("XGBoost认为最重要的特征排名：")
    print(feature_importance)
    
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x='Importance', y='Feature', data=feature_importance)
    # plt.title('XGBoost Feature Importance')
    # fig_path = os.path.join(base_path, "xgboost_feature_importance.png")
    # plt.savefig(fig_path)
    # plt.close()
    # print("特征重要性图已保存。\n")

    # --- 4. 评估模型性能 ---
    print("--- 4. 在独立的测试集和训练集上评估最终模型性能 ---")
    
    # 在测试集上的表现 (最重要的泛化能力指标)
    y_pred_test = best_model.predict(X_test)
    print("\n--- A. 在测试集上的最终评估结果 ---")
    print("混淆矩阵:")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test, y_pred_test, target_names=['错误(Agent)', '正确(LLM)']))
    
    # 在训练集上的表现 (用于检查过拟合)
    y_pred_train = best_model.predict(X_train)
    print("\n--- B. 在训练集上的表现 (用于对比) ---")
    print("混淆矩阵:")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(cm_train)
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_train, y_pred_train, target_names=['错误(Agent)', '正确(LLM)']))

    # 保存详细的评估结果
    evaluation_results = {
        'test_set_results': {
            'confusion_matrix': cm_test.tolist(),
            'classification_report': classification_report(y_test, y_pred_test, target_names=['错误(Agent)', '正确(LLM)'], output_dict=True)
        },
        'train_set_results': {
            'confusion_matrix': cm_train.tolist(),
            'classification_report': classification_report(y_train, y_pred_train, target_names=['错误(Agent)', '正确(LLM)'], output_dict=True)
        },
        'best_parameters': grid_search.best_params_,
        'cross_validation_score': grid_search.best_score_,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    evaluation_path = os.path.join(base_path,"xgboost_evaluation_results.json")
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\n详细评估结果已保存到: {evaluation_path}")
    
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
        print(f"\n模型已保存到: {model_path}")
    #     model_json_path = os.path.join(base_path, "xgboost_Pangu.json")
    # try:
    #     best_model.save_model(model_json_path)
    #     print(f"\n模型已保存到: {model_json_path}")
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
            model.load_model(model_path) # 加载 .json 文件
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

# --- 如何使用 ---
if __name__ == '__main__':
    # 训练模型
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api")
    csv_filename = os.path.join(base_path,"result_dict_five.csv")
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/example/run_one_task/judge_contain_motif/api/xgboost8")
    model_info = train_xgboost_dispatcher(csv_filename)
    
    # # 2. 使用模型进行预测的示例
    # if model_info is not None:
    #     # 示例特征
    #     test_features = {
    #         'cyclomatic_complexity': 5,
    #         'num_nodes': 30,
    #         'edge_locality_score': 2.5,
    #         'ratio_nodes_ge_3': 0.4,
    #         'ratio_nodes_eq_2': 0.3
    #     }
        
    #     # 方法1：直接使用模型信息进行预测
    #     prediction, probability = predict_with_xgboost(test_features, model_info=model_info)
    #     print("\n预测示例1（使用模型信息）:")
    #     print(f"预测结果: {'使用LLM' if prediction == 1 else '使用Agent'}")
    #     print(f"使用LLM的概率: {probability:.2%}")
        
    #     # 方法2：从保存的文件加载模型进行预测
    #     model_path = "/home/hb/LLMDyG_Motif/logs/example/run_one_task/judge_contain_motif/api/xgboost_best_nodes_model.joblib"
    #     prediction, probability = predict_with_xgboost(test_features, model_path=model_path)
    #     print("\n预测示例2（从文件加载模型）:")
    #     print(f"预测结果: {'使用LLM' if prediction == 1 else '使用Agent'}")
    #     print(f"使用LLM的概率: {probability:.2%}")