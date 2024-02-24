# Classification Pipeline for Predicting Mortality of ICU Patients

This project was a homework assignment from Advanced ML course in Armenian Code Academy. The assignment was to implement a Pipeline for ML Classification problem, that should include Preprocessing, Classifier and Pipeline logic. The Dataset for the problem is described here -https://physionet.org/content/challenge-2012/1.0.0/ and the task is to predict the `In-hospital death` label.

For running the pipeline, use python to run `run_pipeline.py` script. Make sure to have all the dependencies installed (these are described in the `requirements.txt `). Here are the flags to use with the script
* ` --test ` - Indicates whether the pipeline should run in test mode or in train mode (train mode if flag is absent). If in train mode, the result of the program is that the fitted preprocessing and classifier models are saved as pickle objects. If in test mode, the pipeline should load already saved preprocessing and classifier models, make predictions on the given dataset and save the predicted probabilities and the threshold in `./predictions.json` file.
* ` --data_path <path> ` - Path to the dataset (.csv file)
* ` --model_path <path> ` - The path to save/load (train/test mode) the classifier model pickle object. If the flag is not present, then the default path (`./saved/model.sav`) is used.
* ` --preprocessor_path <path> ` - The path to save/load (train/test mode) the preprocessing model pickle object. If the flag is not present, then the default path (`./saved/preprocessing.sav`) is used.
* ` --threshold <threshold_value>  ` - The positive label classification threshold. If the flag is not present, then the default value (`threshold_value=0.45`) for the threshold is used.

In addition to the Pipeline, the repository features Jupyter Notebooks that showcase some of the Data Analysis and Model Selection done in the project.

Here is the best classification report obtained with the positive label classification threshold set to 0.45 (threshold choice is based on the recall value for label 1) 
```
AUC - 0.8302115110572863  

              precision    recall  f1-score   support

           0       0.96      0.65      0.77      3446
           1       0.28      0.85      0.42       554

    accuracy                           0.67      4000
   macro avg       0.62      0.75      0.60      4000
weighted avg       0.87      0.67      0.72      4000
```
