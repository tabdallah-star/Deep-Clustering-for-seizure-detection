acc_stat, acc_p_value = wilcoxon(degm_acc_list, baseline_acc_list)
nmi_stat, nmi_p_value = wilcoxon(degm_nmi_list, baseline_nmi_list)
precision_stat, precision_p_value = wilcoxon(degm_precision_list, baseline_precision_list)
recall_stat, recall_p_value = wilcoxon(degm_recall_list, baseline_recall_list)
f1_stat, f1_p_value = wilcoxon(degm_f1_list, baseline_f1_list)

print(f"Wilcoxon signed-rank test for Accuracy: statistic={acc_stat}, p-value={acc_p_value}")
print(f"Wilcoxon signed-rank test for NMI: statistic={nmi_stat}, p-value={nmi_p_value}")
print(f"Wilcoxon signed-rank test for Precision: statistic={precision_stat}, p-value={precision_p_value}")
print(f"Wilcoxon signed-rank test for Recall: statistic={recall_stat}, p-value={recall_p_value}")
print(f"Wilcoxon signed-rank test for F1 Score: statistic={f1_stat}, p-value={f1_p_value}")
