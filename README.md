# Deep-Clustering-for-seizure-detection
1) Read the data:
   
   The publi UCI Epileptic Seizure Dataset can be found online: https://www.kaggle.com/code/maximkumundzhiev/epileptic-seizure-recognition

    The CHU dataset is a local private dataset

2) Slidding Window with 30% overlap on the CHU dataset:
   
   Slidding_Window.mat file 
   
   INPUT: The raw EEG data

   OUTPUT: The series of overlapping data segments
   
   The primary purpose of using sliding windows in EEG data analysis is to break down the continuous EEG signal into smaller, manageable segments for feature 
   extraction and analysis. Overlap in sliding windows is crucial for capturing the dynamic nature of EEG signals and ensuring that no important information is lost.

3) Hyperparameter:
      
   Hyperparameter_Tuning.py

   This section performs hyperparameter tuning using grid search and BIC to find the best parameters for the model.
