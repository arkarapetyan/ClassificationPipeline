# Classification Pipeline for Predicting Mortality of ICU Patients

This project was a homework assignment from Advanced ML course in Armenian Code Academy. The task was to implement a Pipeline for ML Classification problem, that should include Preprocessing, Classifier and Pipeline logic. The Dataset for the problem is described here -https://physionet.org/content/challenge-2012/1.0.0/

Here is the best classification report with the positive label classification threshold set at 0.45 (By maximizing the recall for label 1) 

```
AUC - 0.8302115110572863  

              precision    recall  f1-score   support

           0       0.96      0.65      0.77      3446
           1       0.28      0.85      0.42       554

    accuracy                           0.67      4000
   macro avg       0.62      0.75      0.60      4000
weighted avg       0.87      0.67      0.72      4000
```

Saved preprocessing and model objects are in ```./saved ``` directory.
Threshold can be changed with ```--threshold``` flag.
All other run flags are in accordance with the homework description.
