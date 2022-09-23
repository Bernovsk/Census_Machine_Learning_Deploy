# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The used model was an RandomForestClassifier with 100 estimatores.

## Intended Use
The main objective of the model is predict the financial category of a person, if is above or Below to $50k

## Training Data
The initial data was downloaded from https://archive.ics.uci.edu/ml/datasets/census+income and then preprocessed using the ./ml/data.py script.
After the preprocessing step the data was splited into train and test with 80% and 20% of the initial data respectively.

## Evaluation Data
As mentioned in the training Data, the evaluation was 20% preprocessed held data.

## Metrics
The model metrics was  Precision, Recall, Fbeta, accuracy. In general the model reach 0.846 of accuracy with and std of 0.002 running over a crossvalidation of 5 folders.

## Ethical Considerations
Inside the pipeline, have a validation step in each slice of this data, trying in the best way to discover possible unethical behavior of the model. This evaluation on each slice could be analysed in the following path: /ml/model/sliced_output.txt

## Caveats and Recommendations
In the slice validation we can see that the model have some unethical behavior in some slices, it is important to try to adjust the model.