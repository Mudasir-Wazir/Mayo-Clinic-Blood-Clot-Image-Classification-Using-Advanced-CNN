This project is part of the  Kaggle competition "Mayo Clinic - STRIP AI." 
The primary objective was to classify blood clot origins in ischemic stroke using high-resolution digital pathology images. 
Our task involved developing a model that could accurately differentiate between two major ischemic stroke etiologies: Cardiac Embolism (CE) and Large Artery Atherosclerosis (LAA), making this a classic binary classification problem.

The dataset consisted of train and test .tif images of very high resolution along with their csv files with labels. 
There was also an ‘Other’ additional file with more images, but they were labelled differently than CE and LAA. 


Approaches for image processing:
Approach 1: Traversed the main image in chunks and dynamic strides and resized only informative images to 256*256 and converted to RGB. Used ensemble of 4 pretrained models. But we got an error when submitting to the competition on hidden data set. 
Approach 2: Dividing the original images into smaller patches, the model focuses on localized features and processes each patch more efficiently. Each original high-resolution image is split into smaller, square patches, with dimensions 224x224. We use pyvips library for faster image processing. We were able to submit on hidden test data and got a submission score. But due to time restrictions, we could only use the inception model for this. 

Given the dataset’s relatively limited size, we relied on pre-trained models fine-tuned to optimize classification performance. We experimented with four models:

•	ResNet-50
•	InceptionV3
•	EfficientNetB0
•	Vgg16
For each individual model we enhanced with final dense layer by incorporating batch normalization and dropout for regularization.
Each model was modified with one dense layer tailored for this specific classification. 
Additionally, we conducted fine-tuning experiments, unfreezing the top two layers in ResNet for further optimization; however, this did not significantly impact model performance.

Finally, we used an ensemble of the 4 models to improve the overall performance with chosen weights. 

Model Output and Evaluation

InceptionV3 and Vgg16 performed slightly better with Vgg16 registering the lowest validation loss. 
Evaluated model performance on test data provided by the competition:

- Prediction Analysis: Ensemble predictions showed improved classification probabilities for patients, increasing the confidence in correct classifications.

In the second approach we got validation accuracy of 76% and validation loss of 0.50. 
