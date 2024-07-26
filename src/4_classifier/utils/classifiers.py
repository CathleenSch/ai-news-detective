import itertools
import joblib
import os
import time
import yaml

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

config = yaml.safe_load(open((Path(__file__).parent / './classifier_config.yml').resolve()))

NUM_WORKERS = max(1, int(os.cpu_count() * 0.8))
SAVE_DIRECTORY = config['model.save.directory']

def train_and_evaluate_mlp(params, X_train, X_validate, Y_train, Y_validate):
    hidden_layer_sizes, activation, solver, alpha, learning_rate, max_iter = params
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter
    )
    
    mlp.fit(X_train, Y_train)
    prediction = mlp.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training MLP with: {params} | Score: {score}')
    
    return (params, score, mlp)

def mlp_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 500]
    }
    
    parameter_combinations = list(itertools.product(
        parameter_grid['hidden_layer_sizes'],
        parameter_grid['activation'],
        parameter_grid['solver'],
        parameter_grid['alpha'],
        parameter_grid['learning_rate'],
        parameter_grid['max_iter']
    ))
    
    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for Multilayer Perceptron')
    
    results = []
    for combination in parameter_combinations:
        result = train_and_evaluate_mlp(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        results.append(result)
        progress_bar.update(1)
        
    progress_bar.close()
    
    best_params, best_score, best_model = max(results, key=lambda x: x[1])
    
    print(f'Best Validation Score: {best_score} with best Hyperparameters: {best_params}')

    return best_model

def train_and_evaluate_gb(params, X_train, X_validate, Y_train, Y_validate):
    n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf = params
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    gb.fit(X_train, Y_train)
    prediction = gb.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training GB with: {params} | Score: {score}')

    return (params, score, gb)

def gb_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    parameter_combinations = list(itertools.product(
        parameter_grid['n_estimators'],
        parameter_grid['learning_rate'],
        parameter_grid['max_depth'],
        parameter_grid['min_samples_split'],
        parameter_grid['min_samples_leaf']
    ))
    
    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for Gradient Boosting')
    
    results = []
    for combination in parameter_combinations:
        result = train_and_evaluate_gb(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()
    
    best_params, best_score, best_model = max(results, key=lambda x: x[1])
    
    print(f'Best Validation Score: {best_score} with best Hyperparameters: {best_params}')
    
    return best_model

def train_and_evaluate_rf(params, X_train, X_validate, Y_train, Y_validate):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    rf.fit(X_train, Y_train)
    prediction = rf.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training RF with: {params} | Score: {score}')

    return (params, score, rf)

def rf_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    parameter_combinations = list(itertools.product(
        parameter_grid['n_estimators'],
        parameter_grid['max_depth'],
        parameter_grid['min_samples_split'],
        parameter_grid['min_samples_leaf']
    ))
    
    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for Random Forest')
    
    results = []
    for combination in parameter_combinations:
        result = train_and_evaluate_rf(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()
    
    best_params, best_score, best_model = max(results, key=lambda x: x[1])
    
    print(f'Best Validation Score: {best_score} with best Hyperparameters: {best_params}')
    
    return best_model

def train_and_evaluate_svm(params, X_train, X_validate, Y_train, Y_validate):
    C, gamma, kernel = params

    svm = SVC(
        C=C,
        gamma=gamma,
        kernel=kernel
    )
    
    svm.fit(X_train, Y_train)
    prediction = svm.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training SVM with: {params} | Score: {score}')

    return (params, score, svm)

def svm_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    
    parameter_combinations = list(itertools.product(
        parameter_grid['C'],
        parameter_grid['gamma'],
        parameter_grid['kernel']
    ))
    
    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for Support Vector Machine')
    
    results = []
    for combination in parameter_combinations:
        result = train_and_evaluate_svm(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()
    
    best_params, best_score, best_model = max(results, key=lambda x: x[1])
        
    print(f'Best Validation Score: {best_score} with best Hyperparameters: {best_params}')
        
    return best_model

def train_and_evaluate_lr(params, X_train, X_validate, Y_train, Y_validate):
    C, solver, max_iter = params

    lr = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    
    lr.fit(X_train, Y_train)
    prediction = lr.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training LR with: {params} | Score: {score}')

    return (params, score, lr)

def lr_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'max_iter': [100, 200, 500]
    }
    
    parameter_combinations = list(itertools.product(
        parameter_grid['C'],
        parameter_grid['solver'],
        parameter_grid['max_iter']
    ))

    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for Logistic Regression')
    
    results = []
    for combination in parameter_combinations:
        result = train_and_evaluate_lr(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        results.append(result)
        progress_bar.update(1)
    
    progress_bar.close()
    
    best_params, best_score, best_model = max(results, key=lambda x: x[1])
        
    print(f'Best Validation Score: {best_score} with best Hyperparameters: {best_params}')
        
    return best_model

def train_and_evaluate_knn(params, X_train, X_validate, Y_train, Y_validate):
    n_neighbors, weights, algorithm, p = params

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        p=p
    )    
    
    knn.fit(X_train, Y_train)
    prediction = knn.predict(X_validate)
    score = accuracy_score(Y_validate, prediction)
    print(f'Finished training KNN with: {params} | Score: {score}')

    return (params, score, knn)

def knn_model(X_train, X_validate, Y_train, Y_validate):
    parameter_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1]
    }

    parameter_combinations = list(itertools.product(
        parameter_grid['n_neighbors'],
        parameter_grid['weights'],
        parameter_grid['algorithm'],
        parameter_grid['p']
    ))
    
    progress_bar = tqdm(total=len(parameter_combinations), desc='Hyperparameter Tuning for K Nearest Neighbors')
    
    best_result = ([], 0, None)
    for combination in parameter_combinations:
        result = train_and_evaluate_knn(params=combination, X_train=X_train, X_validate=X_validate, Y_train=Y_train, Y_validate=Y_validate)
        best_result = max((best_result, result), key=lambda x: x[1])
        progress_bar.update(1)
    
    progress_bar.close()
    
    print(f'Best Validation Score: {best_result[1]} with best Hyperparameters: {best_result[0]}')

    return best_result[2]

def train_model(type, X_train, X_validate, Y_train, Y_validate):
    time_start = time.time()
    time_end = time.time()
    model = None
    model_path = ''
    
    if type == 'knn':
        model = knn_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/knn_model.pkl'
    elif type == 'lr':
        model = lr_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/lr_model.pkl'
    elif type == 'svm':
        model = svm_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/svm_model.pkl'
    elif type == 'rf':
        model = rf_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/rf_model.pkl'
    elif type == 'gb':
        model = gb_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/gb_model.pkl'
    elif type == 'mlp':
        model = mlp_model(X_train, X_validate, Y_train, Y_validate)
        time_end = time.time()
        model_path = f'./models/{SAVE_DIRECTORY}/mlp_model.pkl'

    print(f'Training model of type {type} took {time_end-time_start} seconds.')
    joblib.dump(model, model_path)