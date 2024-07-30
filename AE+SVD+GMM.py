import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, losses
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import time

input_shape = (256, 1)
hidden_units = 64
n_clusters = 2
update_interval = 10
pretrain_epochs = 100
ds_name = 'EEG_dataset'

def model_conv(filters_1=128, filters_2=64, kernel_size=3, activation='relu', load_weights=True):
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    input = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=filters_1, kernel_size=kernel_size, strides=2, padding='same', activation=activation, kernel_initializer=init)(input)
    x = layers.Conv1D(filters=filters_2, kernel_size=kernel_size, strides=2, padding='same', activation=activation, kernel_initializer=init)(x)
    flatten_layer = layers.Flatten()
    x = flatten_layer(x)
    x = layers.Dense(units=filters_2, activation=activation)(x)
    x = layers.Dense(units=hidden_units, name='embed', activation=activation)(x)
    h = x
    x = layers.Dense(filters_2, activation=activation)(x)
    x = layers.Dense(filters_1, activation=activation)(x)
    x = layers.Dense(np.prod(input_shape), activation=activation)(x)
    x = layers.Reshape(input_shape)(x)
    output = layers.Concatenate()([h, flatten_layer(x)])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        model.load_weights(f'weight_base_{ds_name}.h5')
        print('model_conv: weights were loaded')
    return model

def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)

def train_base(ds_xx, filters_1, filters_2, kernel_size, activation):
    model = model_conv(filters_1, filters_2, kernel_size, activation, load_weights=False)
    model.compile(optimizer=Adam(learning_rate=0.01), loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
    model.save_weights(f'weight_base_{ds_name}.h5')

def sorted_svd(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    sorted_indices = np.argsort(s)
    sorted_singular_values = s[sorted_indices]
    sorted_singular_vectors = U[:, sorted_indices]
    return sorted_singular_values, sorted_singular_vectors

def get_ACC_NMI(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return acc, nmi

def log_csv(log_str, file_name):
    with open(f'{file_name}.csv', 'a') as f:
        f.write(','.join(log_str) + '\n')

def train(x, y):
    log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv(log_str.split(';'), file_name=ds_name)
    model = model_conv()

    optimizer = Adam()
    loss_value = 0
    kmeans_n_init = 100
    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])
    for ite in range(int(140 * 100)):
        if ite % update_interval == 0:
            H = model(x).numpy()[:, :hidden_units]
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            explained_variance_ratio = np.cumsum(s**2) / np.sum(s**2)
            n_components = np.searchsorted(explained_variance_ratio, 0.95) + 1
            U = U[:, :n_components]
            sorted_indices = np.argsort(s)
            sorted_eigenvalues = s[sorted_indices]
            sorted_eigenvectors = U[:, sorted_indices]

            ans_kmeans = GaussianMixture(n_components=n_clusters, n_init=kmeans_n_init).fit(sorted_eigenvectors)
            yhat = ans_kmeans.predict(sorted_eigenvectors)
            clusters = np.unique(yhat)
            kmeans_n_init = int(ans_kmeans.n_iter_ * 2)

            C = ans_kmeans.means_
            assignment_new = yhat

            w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
            for i in range(len(assignment_new)):
                w[assignment_new[i], assignment[i]] += 1
            ind = linear_sum_assignment(-w)
            temp = np.array(assignment)
            for i in range(n_clusters):
                assignment[temp == ind[1][i]] = i
            n_change_assignment = np.sum(assignment_new != assignment)
            assignment = assignment_new
            loss = np.round(np.mean(loss_value), 5)
            acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))
            print(classification_report(np.array(y), np.array(assignment)))
            log_str = f'iter {ite // update_interval}; acc, nmi, ri = {acc, nmi, loss}; loss:' \
                      f'{loss:.5f}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            print(log_str)
            log_csv(log_str.split(';'), file_name=ds_name)

        if n_change_assignment <= len(x) * 0.005:
            model.save_weights(f'weight_final_l2_{ds_name}.h5')
            print('end')
            break
