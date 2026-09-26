import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

import optuna
from functools import partial
from pathlib import Path
import json
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def create_model(input_dim, hidden_units, activation='relu', alpha=0, out_units=1, out_act='sigmoid', 
                 learning_rate=0.001, momentum=0.9, weight_decay=0, loss='binary_crossentropy'):

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

    return model

def passive_sampling(model, X_unlabelled, n_targets, n_samples):
    selected_indices = np.random.choice(
        X_unlabelled.shape[0],
        size=n_samples,
        replace=False
    )
    return selected_indices

def uncertainty_sampling(model, X_unlabelled, n_classes, n_samples):
    probabilities = model.predict(
        X_unlabelled,
        verbose=0
    )
    selected_indices = None
    if n_classes == 1:
        probabilities = probabilities.ravel()
        uncertainty = np.abs(probabilities - 0.5)
        selected_indices = np.argsort(uncertainty)[:n_samples]
    else:
        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-12),
            axis=1
        )
        selected_indices = np.argsort(-entropy)[:n_samples]
    return selected_indices

# def sasla_sampling(model, X_pool, n_targets, n_samples):
#     S = np.zeros(shape=X_pool.shape[0])
#     probabilities = model.predict(
#         X_pool,
#         verbose=0
#     )
#     hidden_model = tf.keras.Model(
#         inputs=model.input,
#         outputs=model.layers[0].output
#     )
    
#     hidden_values = hidden_model(X_pool, training=False).numpy()
    
#     weights = model.get_weights()
#     hidden_weights = weights[0]
#     output_weights = weights[2]
    
#     for p in range(X_pool.shape[0]):
#         # x_p = X_pool[p]
#         S_p = np.zeros(shape=(n_targets, X_pool.shape[1]))
#         for k in range(n_targets):
#             o_pk = probabilities[p, k]
#             odds = o_pk * (1 - o_pk)

#             # S_pk = np.zeros(X_pool.shape[1])
#             for i in range(X_pool.shape[1]):
#                 total = 0
#                 for j in range(hidden_weights.shape[1]):
#                     h_j = hidden_values[p, j]
#                     w_kj = output_weights[j, k]
#                     w_ji = hidden_weights[i, j]
#                     total += w_kj * w_ji * (1 - h_j) * h_j

#                 S_p[k, i] = odds * total
#         S[p] = np.max(S_p)

#     selected_indices = np.argsort(-S)[:n_samples]
#     return selected_indices

def sasla_sampling(model, X_pool, activation, alpha, n_targets, beta):
    # Model outputs
    probabilities = model(X_pool, training=False).numpy()
    
    hidden_values = model.layers[0](
        X_pool,
        training=False
    ).numpy()
    # hidden_values = hidden_model(X_pool, training=False).numpy()

    # Weights
    weights = model.get_weights()
    hidden_weights = weights[0]    
    output_weights = weights[2]    

    hidden_derivatives = None
    if activation == 'relu':
        hidden_derivative = np.where(
            hidden_values > 0,
            1.0,
            0.0
        )
    elif activation == 'leaky_relu':
        hidden_derivative = np.where(
            hidden_values > 0,
            1.0,
            alpha
        )
    elif activation == 'elu':
        hidden_derivative = np.where(
            hidden_values > 0,
            1.0,
            alpha * np.exp(hidden_values)
        )
    else:
        hidden_derivative = hidden_values * (1 - hidden_values)
    total = np.einsum(
        'pj,jk,ij->pki',
        hidden_derivative,
        output_weights,
        hidden_weights
    )

    output_derivative = probabilities * (1 - probabilities)
    S = total * output_derivative[:, :, np.newaxis]
    S = np.max(np.abs(S), axis=(1, 2))
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

def active_learning_bc(X_pool, y_pool, X_test, y_test, sampling, num_iter=10, unlab_size=0.9, random_state=100, verbose=False,
                       hidden_units=32, activation='relu', alpha=0, balanced=False,
                       learning_rate=0.001, momentum=0.9, weight_decay=0, loss='binary_crossentropy', 
                       ):
    indices = np.arange(len(X_pool))
    # X_labelled, X_unlabelled, y_labelled, y_unlabelled, \
    idx_labelled, idx_unlabelled = train_test_split(
        # X_pool,
        # y_pool,
        indices,
        test_size=unlab_size,
        stratify=y_pool,
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
        model = create_model(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, out_units=1, out_act='sigmoid', 
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
            
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            **fit_kwargs
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob)
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tTraining loss = {train_loss[-1]}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1], 
                        'Accuracy': acc, 'macro F1': f1, 'ROC-AUC': auc, 'time': toc - tic})

        # determine instances that should be labelled
        if i < num_iter - 1:
            selected_indices = sampling(model, X_unlabelled, 1, n_samples)
            idx_labelled, idx_unlabelled = label_data(selected_indices, idx_labelled, idx_unlabelled)

    return history

def active_learning_mc(X_pool, y_pool, X_test, y_test, sampling, num_iter=10, unlab_size=0.9, random_state=100, verbose=False,
                       hidden_units=32, activation='relu', alpha=0, out_units=3, out_act='softmax', balanced=False, 
                       learning_rate=0.001, momentum=0.9, weight_decay=0, loss='sparse_categorical_crossentropy', 
                       ):
    indices = np.arange(len(X_pool))
    # X_labelled, X_unlabelled, y_labelled, y_unlabelled, \
    idx_labelled, idx_unlabelled = train_test_split(
        # X_pool,
        # y_pool,
        indices,
        test_size=unlab_size,
        stratify=y_pool,
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
        # X_labelled = X_pool.iloc[idx_labelled]
        # y_labelled = y_pool.iloc[idx_labelled]
        # X_unlabelled = X_pool.iloc[idx_unlabelled]
        # y_unlabelled = y_pool.iloc[idx_unlabelled]
        X_labelled = X_pool[idx_labelled]
        y_labelled = y_pool[idx_labelled]
        X_unlabelled = X_pool[idx_unlabelled]
        y_unlabelled = y_pool[idx_unlabelled]

        # compile model
        model = create_model(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, out_units=out_units, out_act=out_act, 
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
            
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            **fit_kwargs
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tAccuracy = {acc}\n"
                  f"\tmacro F1 = {f1}\n"
                  f"\tAUC = {auc}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1],
                        'Accuracy': acc, 'macro F1': f1, 'ROC-AUC': auc, 'time': toc-tic})

        # determine instances that should be labelled
        if i < num_iter - 1:
            selected_indices = sampling(model, X_unlabelled, out_units, n_samples)
            idx_labelled, idx_unlabelled = label_data(selected_indices, idx_labelled, idx_unlabelled)

    return history

def sasla_bc(X_pool, y_pool, X_test, y_test, num_iter=10, unlab_size=0.9, random_state=100, verbose=False,
                       hidden_units=32, activation='relu', alpha=0, balanced=False,
                       learning_rate=0.001, momentum=0.9, weight_decay=0, loss='binary_crossentropy', beta=0.9
                       ):
    indices = np.arange(len(X_pool))
    idx_labelled = indices.copy()
    
    input_dim = X_pool.shape[1]
    # n_samples = X_pool.shape[0]
    # n_reduce = np.ceil(n_samples * unlab_size) // (num_iter-1)
    # excess = np.ceil(n_samples * unlab_size) - n_reduce * (num_iter-1)
    epochs = 200 // num_iter
    
    history = []
    tic = time()
    for i in range(num_iter):

        # update pool of labelled data
        X_labelled = X_pool[idx_labelled]
        y_labelled = y_pool[idx_labelled]

        #compile model
        model = create_model(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, out_units=1, out_act='sigmoid', 
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            **fit_kwargs
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob)
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tAccuracy = {acc}\n"
                  f"\tmacro F1 = {f1}\n"
                  f"\tAUC = {auc}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1], 
                        'Accuracy': acc, 'macro F1': f1, 'ROC-AUC': auc, 'time': toc - tic})

        # # reduce subset size
        # n_samples -= n_reduce
        # if i == 0:
        #     n_samples -= excess

        # determine instances that should be labelled
        if i < num_iter - 1:
            idx_labelled = sasla_sampling(model, X_pool, activation, alpha, 1, beta)
            if len(idx_labelled) < (1 - unlab_size) *X_pool.shape[0]:
                break

    return history

def sasla_mc(X_pool, y_pool, X_test, y_test, num_iter=10, unlab_size=0.9, random_state=100, verbose=False,
                       hidden_units=32, activation='relu', alpha=0, out_units=3, out_act='softmax', balanced=False, 
                       learning_rate=0.001, momentum=0.9, weight_decay=0, loss='sparse_categorical_crossentropy', beta=0.9
                       ):
    indices = np.arange(len(X_pool))
    idx_labelled = indices.copy()

    input_dim = X_pool.shape[1]
    # n_samples = X_pool.shape[0]
    # n_reduce = n_samples * unlab_size // (num_iter-1)
    # excess = n_samples * unlab_size - n_reduce * (num_iter-1)
    epochs = 200 // num_iter
    history = []
    tic = time()
    for i in range(num_iter):
        # update pool of labelled data
        X_labelled = X_pool[idx_labelled]
        y_labelled = y_pool[idx_labelled]

        # compile model
        model = create_model(input_dim=input_dim, hidden_units=hidden_units, activation=activation, alpha=alpha, out_units=out_units, out_act=out_act, 
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
            
        iter_hist = model.fit(
            X_labelled,
            y_labelled,
            **fit_kwargs
        )
        train_loss = iter_hist.history['loss']

         # Evaluate
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
        toc = time()

        if verbose:
            print(f"Iteration {i+1}: {idx_labelled.shape[0]} labelled instances\n"
                  f"\tAccuracy = {acc}\n"
                  f"\tmacro F1 = {f1}\n"
                  f"\tAUC = {auc}\n"
            )
        history.append({'Labelled': idx_labelled.shape[0], 'cumulative epochs': epochs * (i + 1), 'Training loss': train_loss[-1],
                        'Accuracy': acc, 'macro F1': f1, 'ROC-AUC': auc, 'time': toc - tic})

        # # reduce subset size
        # n_samples -= n_reduce
        # if i == 0:
        #     n_samples -= excess

        # determine instances that should be labelled
        if i < num_iter - 1:
            idx_labelled = sasla_sampling(model, X_pool, activation, alpha, out_units, beta)
            if len(idx_labelled) < (1 - unlab_size) *X_pool.shape[0]:
                break

    return history

def balanced_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))
    return class_weights

def obj(trial, X_train, y_train, out_units=1, out_act='sigmoid', loss="binary_crossentropy", k=5, activation='relu', balanced=False, random_state=42):
    params = {
        "hidden_layer": trial.suggest_int(
            "hidden_layer", 32, 128
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
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

    cv = StratifiedKFold(
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
            
        model.fit(
            X_train_split,
            y_train_split,
            **fit_kwargs
        )

        if out_units == 1:
            y_prob = model(X_val_split, training=False).numpy().ravel()
            score = tf.keras.losses.binary_crossentropy(
                y_val_split,
                y_prob
            )
            scores.append(np.mean(score))
        else:
            y_prob = model(X_val_split, training=False).numpy()
            score = tf.keras.losses.sparse_categorical_crossentropy(
                y_val_split,
                y_prob
            )
            scores.append(np.mean(score))

        avg_so_far = np.mean(scores)

        trial.report(avg_so_far, step=len(scores))
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    avg_score = np.mean(scores)
    return avg_score

def tune_params(X_train, y_train, loss="binary_crossentropy", out_units=1, out_act='sigmoid', k=5, activation='relu', balanced=False, random_state=42, 
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
        balanced=balanced,
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

def cv_classification(X_train, y_train, out_units=1, out_act='sigmoid', loss="binary_crossentropy", k=5, random_state=42, 
                  hidden_units=32, activation='relu', alpha=0, 
                  learning_rate=0.01, momentum=0, weight_decay=0, balanced=False):

    y_pred = None
    
    cv = StratifiedKFold(
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
        if balanced:
            fit_kwargs["class_weight"] = balanced_weights(
                y_train_split
            )
            
        model.fit(
            X_train_split,
            y_train_split,
            **fit_kwargs
        )

        score = {}
        if out_units == 1:
            y_prob = model(X_val_split, training=False).numpy().ravel()
            y_pred = (y_prob >= 0.5).astype(int)
            auc = roc_auc_score(y_val_split, y_prob)
            score['auc'] = auc
        else:
            y_prob = model(X_val_split, training=False).numpy()
            y_pred = np.argmax(y_prob, axis=1)
            auc = roc_auc_score(
                y_val_split,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
            score['auc'] = auc
            
        acc = accuracy_score(y_val_split, y_pred)
        f1 = f1_score(y_val_split, y_pred, average='macro')
        score['acc'] = acc
        score['f1'] = f1
        scores.append(score)
    
    avg_score = {
        metric: np.mean([score[metric] for score in scores])
        for metric in scores[0]
    }
    return avg_score