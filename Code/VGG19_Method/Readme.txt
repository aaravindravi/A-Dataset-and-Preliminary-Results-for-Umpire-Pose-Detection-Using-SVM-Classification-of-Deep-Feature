Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

Model 1 - For Umpire versus Non-Umpire Classification
Model 2 - 4 Classes of Umpire Actions - Six, No Ball, Out, Wide Classification

1. Run feature extraction as Step 1 on all datasets
  a. vgg19_feature_extraction_cricket_model1.py
  b. vgg19_feature_extraction_cricket_model2.py

At the end of this step you should have the features extracted into individual numpy files for both methods.

2. Train the SVM classifiers on both sets of features
This step is performed to train the the classifer and save the model for use in the video summarization step.
  a. vgg19_classifier_model1.py
  b. vgg19_classifier_model2.py

3. Run the video summarization on test video

Dataset: https://drive.google.com/drive/folders/1ljDIz69mJqDzBlUxABP0c34NgdynzWPX
