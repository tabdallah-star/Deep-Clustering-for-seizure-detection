param_grid = {
    'filters_1': [64, 128, 256],
    'filters_2': [32, 64, 128],
    'kernel_size': [2, 3, 5],
    'activation': ['relu', 'leaky_relu']
}

best_params = None
best_bic = float('inf')

for filters_1 in param_grid['filters_1']:
    for filters_2 in param_grid['filters_2']:
        for kernel_size in param_grid['kernel_size']:
            for activation in param_grid['activation']:
                train_base(x_train, filters_1, filters_2, kernel_size, activation)
                model = model_conv(filters_1, filters_2, kernel_size, activation)
                H = model(x_train).numpy()[:, :hidden_units]
                U, s, Vt = np.linalg.svd(H, full_matrices=False)
                explained_variance_ratio = np.cumsum(s**2) / np.sum(s**2)
                n_components = np.searchsorted(explained_variance_ratio, 0.95) + 1
                U = U[:, :n_components]
                sorted_indices = np.argsort(s)
                sorted_eigenvalues = s[sorted_indices]
                sorted_eigenvectors = U[:, sorted_indices]

                gmm = GaussianMixture(n_components=n_clusters, n_init=100).fit(sorted_eigenvectors)
                bic = gmm.bic(sorted_eigenvectors)

                if bic < best_bic:
                    best_bic = bic
                    best_params = {
                        'filters_1': filters_1,
                        'filters_2': filters_2,
                        'kernel_size': kernel_size,
                        'activation': activation
                    }

print(f"Best parameters: {best_params}")
print(f"Best BIC: {best_bic}")


train_base(x_train, best_params['filters_1'], best_params['filters_2'], best_params['kernel_size'], best_params['activation'])
train(x_train, y_train)
