degm_acc_list = []
degm_nmi_list = []
baseline_acc_list = []
baseline_nmi_list = []

degm_precision_list = []
degm_recall_list = []
degm_f1_list = []
baseline_precision_list = []
baseline_recall_list = []
baseline_f1_list = []

for _ in range(5):
    # Train and evaluate DEGM model
    train_base(x_train, best_params['filters_1'], best_params['filters_2'], best_params['kernel_size'], best_params['activation'])
    model = model_conv(best_params['filters_1'], best_params['filters_2'], best_params['kernel_size'], best_params['activation'])
    H = model(x_test).numpy()[:, :hidden_units]
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    explained_variance_ratio = np.cumsum(s**2) / np.sum(s**2)
    n_components = np.searchsorted(explained_variance_ratio, 0.95) + 1
    U = U[:, :n_components]
    sorted_indices = np.argsort(s)
    sorted_eigenvalues = s[sorted_indices]
    sorted_eigenvectors = U[:, sorted_indices]

    gmm = GaussianMixture(n_components=n_clusters, n_init=100).fit(sorted_eigenvectors)
    yhat = gmm.predict(sorted_eigenvectors)
    acc, nmi = get_ACC_NMI(y_test, yhat)
    degm_acc_list.append(acc)
    degm_nmi_list.append(nmi)

    precision = precision_score(y_test, yhat, average='weighted')
    recall = recall_score(y_test, yhat, average='weighted')
    f1 = f1_score(y_test, yhat, average='weighted')
    degm_precision_list.append(precision)
    degm_recall_list.append(recall)
    degm_f1_list.append(f1)

    # Train and evaluate baseline model (i.e.,GMM)
    baseline_model = GaussianMixture(n_components=n_clusters, n_init=100).fit(x_train)
    yhat_baseline = baseline_model.predict(x_test)
    acc_baseline, nmi_baseline = get_ACC_NMI(y_test, yhat_baseline)
    baseline_acc_list.append(acc_baseline)
    baseline_nmi_list.append(nmi_baseline)

    precision_baseline = precision_score(y_test, yhat_baseline, average='weighted')
    recall_baseline = recall_score(y_test, yhat_baseline, average='weighted')
    f1_baseline = f1_score(y_test, yhat_baseline, average='weighted')
    baseline_precision_list.append(precision_baseline)
    baseline_recall_list.append(recall_baseline)
    baseline_f1_list.append(f1_baseline)

# Calculate mean and standard deviation
degm_acc_mean = np.mean(degm_acc_list)
degm_acc_std = np.std(degm_acc_list)
degm_nmi_mean = np.mean(degm_nmi_list)
degm_nmi_std = np.std(degm_nmi_list)
degm_precision_mean = np.mean(degm_precision_list)
degm_precision_std = np.std(degm_precision_list)
degm_recall_mean = np.mean(degm_recall_list)
degm_recall_std = np.std(degm_recall_list)
degm_f1_mean = np.mean(degm_f1_list)
degm_f1_std = np.std(degm_f1_list)

baseline_acc_mean = np.mean(baseline_acc_list)
baseline_acc_std = np.std(baseline_acc_list)
baseline_nmi_mean = np.mean(baseline_nmi_list)
baseline_nmi_std = np.std(baseline_nmi_list)
baseline_precision_mean = np.mean(baseline_precision_list)
baseline_precision_std = np.std(baseline_precision_list)
baseline_recall_mean = np.mean(baseline_recall_list)
baseline_recall_std = np.std(baseline_recall_list)
baseline_f1_mean = np.mean(baseline_f1_list)
baseline_f1_std = np.std(baseline_f1_list)

print(f"DEGM Model - Accuracy: {degm_acc_mean:.4f} ± {degm_acc_std:.4f}")
print(f"DEGM Model - NMI: {degm_nmi_mean:.4f} ± {degm_nmi_std:.4f}")
print(f"DEGM Model - Precision: {degm_precision_mean:.4f} ± {degm_precision_std:.4f}")
print(f"DEGM Model - Recall: {degm_recall_mean:.4f} ± {degm_recall_std:.4f}")
print(f"DEGM Model - F1 Score: {degm_f1_mean:.4f} ± {degm_f1_std:.4f}")
print(f"Baseline Model - Accuracy: {baseline_acc_mean:.4f} ± {baseline_acc_std:.4f}")
print(f"Baseline Model - NMI: {baseline_nmi_mean:.4f} ± {baseline_nmi_std:.4f}")
print(f"Baseline Model - Precision: {baseline_precision_mean:.4f} ± {baseline_precision_std:.4f}")
print(f"Baseline Model - Recall: {baseline_recall_mean:.4f} ± {baseline_recall_std:.4f}")
print(f"Baseline Model - F1 Score: {baseline_f1_mean:.4f} ± {baseline_f1_std:.4f}")
