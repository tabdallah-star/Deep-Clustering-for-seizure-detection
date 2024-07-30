# Deep-Clustering-for-epileptic-seizure-detection-paper
# 1) Read the data:
   The CHU dataset is a local private dataset 
   
   The publi UCI Epileptic Seizure Dataset can be found online: https://www.kaggle.com/code/maximkumundzhiev/epileptic-seizure-recognition


# 2) Slidding Window with 30% overlap on the CHU dataset:
   
  File: Slidding_Window.mat 
   
   INPUT: The raw EEG data

   OUTPUT: The series of overlapping data segments
   
   The primary purpose of using sliding windows in EEG data analysis is to break down the continuous EEG signal into smaller, manageable segments for feature 
   extraction and analysis. Overlap in sliding windows is crucial for capturing the dynamic nature of EEG signals and ensuring that no important information is lost.

# 3) The model

   File: AE+SVD+GMM.py

   INPUT: The series of overlapping data segments

   OUTPUT: The final output is a set of cluster labels assigned to each data point based on the GMM clustering.
   
   This section trains the autoencoder (AE), performs Singular Value Decomposition (SVD) on the hidden layer's output, and applies Gaussian Mixture Model (GMM) for clustering.

# 4) Hyperparameter:
      
  File: Hyperparameter_Tuning.py 

  INPUT: A defined set of parameters that can be adjusted, such as learning rate, number of layers, kernel size, etc

  OUTPUT: The combination of hyperparameters that yields the best performance on the validation set.

  This section performs hyperparameter tuning using grid search and BIC to find the best parameters for the model.

# 5) Evaluation Metrics 

  File: Evaluation_Metrics.py

  INPUT: A dataset containing both input features and corresponding ground truth labels.

  OUTPUT: The various performance metrics.

  This section evaluates the model's performance by running it multiple times and computing accuracy, NMI, precision, recall, and F1 score.

  # 6) P-value

  File: Wilcoxon_Signed-Rank_Test.py

  INPUT: A matrix or data frame containing the performance metrics (accuracy, NMI, precision, recall, F1-score) for each of the 5 runs. Each row represents a run, and each column represents a metric.

  OUTPUT: For each performance metric, a p-value indicating the probability of observing the observed differences in performance by chance if there were no true differences between the runs.

 This section compares the results of the DEGM model after 5 runs using the Wilcoxon signed-rank test.
  
