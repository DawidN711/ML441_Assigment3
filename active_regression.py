import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

import optuna
from functools import partial
from pathlib import Path
import json
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

def create_regmodel(input_dim, hidden_units, activation='relu', alpha=0, dropout_rate=0.2,  
                 learning_rate=0.001, momentum=0.9, weight_decay=0, loss='mse'):

    model_layers = [
        layers.Input(shape=(input_dim,))
    ]

    if activation == "leaky_relu":
        model_layers.append(layers.Dense(hidden_units))
        model_layers.append(layers.LeakyReLU(negative_slope=alpha))

    elif activation == "elu":
        model_layers.append(layers.Dense(hidden_units))
        model_layers.append(layers.ELU(alpha=alpha))

    else:
        model_layers.append(layers.Dense(hidden_units, activation=activation))

    model_layers.append(layers.Dropout(dropout_rate))
    model_layers.append(layers.Dense(1, activation='linear'))
    model = tf.keras.Sequential(model_layers)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        ),
        loss=loss
    )

    return model

def passive_sampling(model, X_unlabelled, n_samples):
    selected_indices = np.random.choice(
        X_unlabelled.shape[0],
        size=n_samples,
        replace=False
    )
    return selected_indices

def uncertainty_sampling_reg(model, X_unlabelled, n_samples):
    predictions = np.array([
        model(X_unlabelled, training=True).numpy().ravel()
        for _ in range(20)
    ])
    
    uncertainty = np.var(predictions, axis=0)
    
    selected_indices = np.argsort(-uncertainty)[:n_samples]
    return selected_indices

def sasla_sampling_reg(model, X_pool, activation, alpha, n_samples):
    X = tf.convert_to_tensor(X_pool, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X)
        outputs = model(X, training=False)
    
    J = tape.batch_jacobian(outputs, X).numpy()
    S = np.max(np.abs(J), axis=(1, 2))

    s_mean = np.mean(S)
    thres = (1 - beta) * s_mean
    # selected_indices = np.argsort(-S)[:n_samples]
    selected_indices = np.where(S > thres)[0]
    return selected_indices

def label_data(selected_indices, labelled, unlabelled):
    selected_pool_indices = unlabelled[selected_indices]
    new_labelled = np.concatenate([labelled, selected_pool_indices], axis=0)
    new_unlabelled = np.setdiff1d(
        unlabelled,
        selected_pool_indices
    )

    return new_labelled, new_unlabelled

def active_learning_reg(X_pool, y_pool, X_test, y_test, sampling, num_iter=10, unlab_size=0.9, random_state=100, verbose=False,
                       hidden_units=32, activation='relu', alpha=0, dropout_rate=0.2,
                       learning_rate=0.001, momentum=0.9, weight_decay=0, loss='mse', 
                       ):
    indices = np.arange(len(X_pool))
    # X_labelled, X_unlabelled, y_labelled, y_unlabelled, \
    idx_labelled, idx_unlabelled = train_test_split(
        # X_pool,
        # y_pool,
        indices,
        test_size=unlab_size,
        random_state=random_state
    )

    input_dim = X_pool.shape[1]
    n_samples = len(idx_unlabelled) // (num_iter-1)
    epochs = 200 // num_iter
    
    history = []
    tic = time()
    for i in range(num_iter):
        if i == num_iter - 2:
            n_samples = idx_unlabelled.shape[0]

        # update pool of labelled data
        X_labelled = X_pool[idx_labelled]
        y_labelled = y_pool[idx_labelled]
        X_unlabelled = X_pool[idx_unlabelled]
        y_unlabelled = y_pool[idx_unlabelled]

        #compile model
        model = create_regmodel(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, dropout_rate=dropout_rate, 
                                out_units=1, out_act='sigmoid', 
                                learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, loss=loss)

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=5,
        #     restore_best_weights=True
        # )

        # train model on labelled data
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            epochs=epochs,
            batch_size=32,
            # validation_split=0.15,
            # callbacks=[early_stopping],
            verbose=0
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_pred = model(X_test, training=False).numpy().ravel()
            
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tRMSE = {rmse}\n"
                  f"\tMAPE = {mape}\n"
                  f"\tR^2 = {r2}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1], 
                        'rmse': rmse, 'mape': mape, 'R_squared': r2, 'time': toc - tic}})

        # determine instances that should be labelled
        if i < num_iter - 1:
            selected_indices = sampling(model, X_unlabelled, n_samples)
            idx_labelled, idx_unlabelled = label_data(selected_indices, idx_labelled, idx_unlabelled)

    return history

def sasla_reg(X_pool, y_pool, X_test, y_test, num_iter=10, random_state=100, verbose=False,
               hidden_units=32, activation='relu', alpha=0, dropout_rate=0.2,
               learning_rate=0.001, momentum=0.9, weight_decay=0, loss='mse', beta=0.9):
    
    indices = np.arange(len(X_pool))
    idx_labelled = indices.copy()

    input_dim = X_pool.shape[1]
    # n_samples = X_pool.shape[0]
    # n_reduce = n_samples * unlab // (num_iter-1)
    # excess = n_samples * unlab - n_reduce * (num_iter-1)
    epochs = 200 // num_iter
    history = []
    tic = time()
    for i in range(num_iter):

        # update pool of labelled data
        X_labelled = X_pool[idx_labelled]
        y_labelled = y_pool[idx_labelled]

        #compile model
        model = create_regmodel(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, dropout_rate=dropout_rate,
                                out_units=1, out_act='sigmoid', 
                                learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, loss=loss)

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=5,
        #     restore_best_weights=True
        # )
        
        # train model on labelled data
        fit_kwargs = {
            "epochs": epochs,
            "batch_size": 32,
            # "validation_split": 0.15,
            # "callbacks": [early_stopping],
            "verbose": 0
        }
            
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            **fit_kwargs
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_pred = model(X_test, training=False).numpy().ravel()
            
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tAccuracy = {acc}\n"
                  f"\tmacro F1 = {f1}\n"
                  f"\tAUC = {auc}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1], 
                        'rmse': rmse, 'mape': mape, 'R_squared': r2, 'time': toc - tic}})

        # determine instances that should be labelled
        if i < num_iter - 1:
            idx_labelled = sasla_sampling_reg(model, X_pool, activation, alpha, beta)

        # n_samples -= n_reduce
        # if i == 0:
        #     n_samples -= excess

    return history

def obj(trial, X_train, y_train, out_units=1, out_act='sigmoid', loss="mse", k=5, activation='relu', dropout_rate=0.2, random_state=42):
    params = {
        "hidden_layer": trial.suggest_int(
            "hidden_layer", 32, 128
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "momentum": trial.suggest_float(
            "momentum",
            0.0,
            0.5
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True
        )
        
    }

    if activation == 'leaky_relu':
        params['alpha'] = trial.suggest_float(
           "leaky_relu_alpha",
            0.001,
            0.3,
            log=True
        )
    elif activation == "elu":
        params['alpha'] = trial.suggest_float(
            "elu_alpha",
            0.1,
            2.0
        )

    cv = KFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )
    scaler = StandardScaler()
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        tf.keras.backend.clear_session()
        
        X_train_split = X_train[train_idx]
        X_train_split = scaler.fit_transform(X_train_split)
        X_val_split = X_train[val_idx]
        X_val_split = scaler.transform(X_val_split)

        y_train_split = y_train[train_idx]
        y_val_split = y_train[val_idx]

        model_layers = [
            layers.Input(shape=(X_train.shape[1],))
        ]

        if activation == "leaky_relu":
            model_layers.append(layers.Dense(params["hidden_layer"]))
            model_layers.append(layers.LeakyReLU(negative_slope=params["alpha"]))

        elif activation == "elu":
            model_layers.append(layers.Dense(params["hidden_layer"]))
            model_layers.append(layers.ELU(alpha=params["alpha"]))

        else:
            model_layers.append(layers.Dense(params["hidden_layer"], activation=activation))

        model_layers.append(layers.Dropout(dropout_rate))
        model_layers.append(layers.Dense(out_units, activation=out_act))
        model = tf.keras.Sequential(model_layers)
    
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=params['learning_rate'],
                momentum=params['momentum'],
                weight_decay=params['weight_decay']
            ),
            loss=loss
        )

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
        fit_kwargs = {
            "epochs": 75,
            "batch_size": 32,
            "validation_split": 0.15,
            "callbacks": [es],
            "verbose": 0
        }
            
        history = model.fit(
            X_train_split,
            y_train_split,
            **fit_kwargs
        )

        train_loss = np.asarray(history.history["loss"])
        if not np.all(np.isfinite(train_loss)):
            return float("inf")

        y_pred = model(X_val_split, training=False).numpy()
        if not np.all(np.isfinite(y_pred)):
            return float("inf")
        score = mean_squared_error(y_val_split, y_pred)
        scores.append(score)

        avg_so_far = np.mean(scores)

        trial.report(avg_so_far, step=len(scores))
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    avg_score = np.mean(scores)
    return avg_score

def tune_params(X_train, y_train, loss="mse", out_units=1, out_act='linear', k=5, activation='relu', dropout_rate=0.2, random_state=42, 
                n_trials=100, export=False, file='params.json'):

    objective = partial(
        obj,
        X_train=X_train,
        y_train=y_train,
        out_units=out_units,
        out_act=out_act,
        loss=loss,
        k=k,
        activation=activation,
        dropout_rate=dropout_rate,
        random_state=random_state
    )
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)
    
    print(study.best_params)
    print(study.best_value)

    best_found = {}
    for key, val in study.best_params.items():
        best_found[key] = val
        
    best_found['score'] = study.best_value
    
    if export:
        path = Path(file)
        if path.is_file():
            old_params = retrieve_params(file)
            if old_params['score'] > best_found['score']:
                with open(file, 'w') as f:
                    json.dump(best_found, f, indent=4)
        else:
            with open(file, 'w') as f:
                json.dump(best_found, f, indent=4)
    
    return best_found

def retrieve_params(file='params.json'):
    with open(file, "r") as f:
        return json.load(f)

def cv_regression(X_train, y_train, out_units=1, out_act='linear', loss="mse", k=5, random_state=42, 
                  hidden_units=32, activation='relu', alpha=0, dropout_rate=0.2, 
                  learning_rate=0.01, momentum=0, weight_decay=0):

    y_pred = None
    
    cv = KFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )
    scaler = StandardScaler()
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        tf.keras.backend.clear_session()
        
        X_train_split = X_train[train_idx]
        X_train_split = scaler.fit_transform(X_train_split)
        X_val_split = X_train[val_idx]
        X_val_split = scaler.transform(X_val_split)

        y_train_split = y_train[train_idx]
        y_val_split = y_train[val_idx]

        model_layers = [
            layers.Input(shape=(X_train.shape[1],))
        ]

        if activation == "leaky_relu":
            model_layers.append(layers.Dense(hidden_units))
            model_layers.append(layers.LeakyReLU(negative_slope=alpha))

        elif activation == "elu":
            model_layers.append(layers.Dense(hidden_units))
            model_layers.append(layers.ELU(alpha=alpha))

        else:
            model_layers.append(layers.Dense(hidden_units, activation=activation))

        model_layers.append(layers.Dropout(dropout_rate))
        model_layers.append(layers.Dense(out_units, activation=out_act))
        model = tf.keras.Sequential(model_layers)
    
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            ),
            loss=loss
        )

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        fit_kwargs = {
            "epochs": 75,
            "batch_size": 32,
            "validation_split": 0.15,
            "callbacks": [es],
            "verbose": 0
        }
            
        model.fit(
            X_train_split,
            y_train_split,
            **fit_kwargs
        )


        y_pred = model(X_val_split, training=False).numpy().ravel()
            
        rmse = root_mean_squared_error(y_val_split, y_pred)
        mape = mean_absolute_percentage_error(y_val_split, y_pred)
        r2 = r2_score(y_val_split, y_pred)
        scores.append({'rmse': rmse, 'mape': mape, 'R_squared': r2})
    
    avg_score = {
        metric: np.mean([score[metric] for score in scores])
        for metric in scores[0]
    }
    return avg_score