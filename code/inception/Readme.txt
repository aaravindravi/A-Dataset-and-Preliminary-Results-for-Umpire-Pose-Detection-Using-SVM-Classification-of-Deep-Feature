Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

Model 1 - For Umpire versus Non-Umpire Classification
Model 2 - 5 Classes of Umpire Actions - Six, No Ball, Out, Wide, No Action Classification

***************
NOTE - Initial Steps
OPTION 1 : Create folder names same as the class names and copy the images into respective folders
OPTION 2 : Read and use the class names from the filename directly in the code
***************

1. Run feature extraction as Step 1 on all datasets
  a. inceptionv3_feature_extraction_model1.py
  b. inceptionv3_feature_extraction_model2.py

At the end of this step you should have the features extracted into individual numpy files for both methods.

2. Train the SVM classifiers on both sets of features
This step is performed to train the the classifer and save the model for use in the video summarization step.
  a. inceptionv3_classifier_model1.py
  b. inceptionv3_classifier_model2.py

3. Run the video summarization on test video

Dataset: https://drive.google.com/drive/folders/1ljDIz69mJqDzBlUxABP0c34NgdynzWPX
