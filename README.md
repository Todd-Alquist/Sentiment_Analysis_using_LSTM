# Movie Review Sentiment Analysis Using LSTM

## Overview
This project involves building an LSTM (Long Short-Term Memory) model to perform sentiment analysis on movie reviews. The goal is to classify reviews as positive or negative, leveraging deep learning techniques to handle the complexities of natural language processing.

## Dataset
The dataset used for this project contains movie reviews along with their sentiment labels. The reviews are preprocessed to convert text into sequences that can be fed into the LSTM model for training and prediction.

### Dataset Details
- **Source:**  [Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Contents:** The dataset contains two columns - 'Review' (the text of the movie review) and 'Sentiment' (the sentiment label, where 1 indicates a positive review and 0 indicates a negative review).



## Objectives
- **Sentiment Classification:** Develop an LSTM model to accurately classify movie reviews as positive or negative.
- **Deep Learning Techniques:** Utilize LSTM networks to capture the sequential nature of text data and improve classification accuracy.
- **Performance Evaluation:** Assess model performance using appropriate evaluation metrics and visualize results.

## Methodology
1. **Data Collection and Preprocessing:**
   - Collected movie review data with sentiment labels.
   - Tokenized and padded sequences to prepare the data for the LSTM model.
   - Split the dataset into training and testing sets.

2. **Model Development:**
   - Built an LSTM model using Keras and TensorFlow.
   - Configured the model with embedding layers, LSTM layers, and dense layers for classification.
   - Compiled the model with appropriate loss functions and optimizers.

3. **Model Training:**
   - Trained the LSTM model on the training data.
   - Monitored training performance using validation data and adjusted hyperparameters accordingly.

4. **Model Evaluation:**
   - Evaluated the model using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
   - Visualized model performance using confusion matrices and classification reports.

## Results
- **Model Performance:** The LSTM model achieved high accuracy in classifying movie reviews as positive or negative.
- **Insights:** Identified key patterns in the text data that contribute to sentiment classification.

### Confusion Matrix
The confusion matrix below illustrates the performance of our model in terms of true positives, true negatives, false positives, and false negatives.

![image](https://github.com/user-attachments/assets/14c13762-eec4-4634-8fd7-9265e2148efb)

### Classification Report
The classification report below provides a detailed breakdown of precision, recall, and F1-score for each class.

![image](https://github.com/user-attachments/assets/0c487ecc-0aaa-4a0c-8443-a1e5f952aa09)


## Acknowledgements
- Kaggle for providing the dataset.
- All contributors and the open-source community.



