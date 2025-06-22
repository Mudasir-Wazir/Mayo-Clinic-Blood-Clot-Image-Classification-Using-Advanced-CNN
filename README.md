# Mayo Clinic Blood Clot Image Classification Using Advanced CNN

## Project Overview

This project aims to develop a robust and accurate deep learning system for the classification of blood clot images, leveraging advanced Convolutional Neural Network (CNN) architectures. The system is designed to assist medical professionals in the diagnosis of blood clot conditions by automating image-based analysis, thus reducing manual effort and increasing diagnostic accuracy. The project is based on datasets provided by the Mayo Clinic and focuses on distinguishing between different types of blood clots in medical images.

---

## Methodology

1. **Data Acquisition & Preprocessing**
   - Collected a curated dataset of blood clot images, annotated by medical experts.
   - Performed data cleaning, normalization, and augmentation (rotation, flipping, scaling) to increase dataset diversity and prevent overfitting.
   - Split the dataset into training, validation, and test sets (typically 70/15/15%).

2. 🖼️ **Approaches for Image Processing**

      🔹 Approach 1
      - The main image was processed in chunks using dynamic strides.
      - Only informative regions were resized to 256×256 and converted to RGB format.
      - An ensemble of four pretrained models was used for prediction.
      - However, this approach encountered an error during submission to the hidden test set in the competition.
      
      🔹 Approach 2
      - High-resolution images were split into smaller square patches (224×224) to capture localized features effectively.
      - The `pyvips` library enabled faster and efficient image preprocessing.
      - This method successfully processed and submitted predictions for the hidden test set.
      - Due to time limitations, only the Inception model was implemented for this approach.


3. **Model Selection & Architecture**
   - Explored and implemented several CNN architectures, including custom CNNs and transfer learning from pretrained models (e.g., ResNet, VGG, Inception).
   - Fine-tuned hyperparameters (learning rate, batch size, optimizer, dropout) using the validation set.
   - For each individual model we enhanced with final dense layer by incorporating batch normalization and dropout for regularization. Each model was modified with one dense layer tailored for this specific classification. Additionally, we conducted fine-tuning experiments, unfreezing the top two layers in ResNet for further optimization; however, this did not significantly impact model performance

4. **Training & Evaluation**
   - Trained models on GPU-accelerated environments for faster convergence.
   - Used categorical cross-entropy loss and accuracy as primary metrics.
   - Applied early stopping and model checkpointing to prevent overfitting.

5. **Testing & Results Analysis**
   - Evaluated the trained models on the reserved test set.
   - Generated confusion matrices, ROC curves, and calculated precision, recall, and F1-score.
   - Performed error analysis to identify common misclassifications.

---

## Key Features

- **Automated Image Classification:** End-to-end pipeline for classifying blood clot images.
- **Advanced CNN Architectures:** Utilizes state-of-the-art deep learning models for improved accuracy.
- **Data Augmentation:** Reduces overfitting and enhances model generalization.
- **Visualization Tools:** Provides visual insights into model performance (confusion matrix, ROC curves).


---

## Models Used

- **Custom CNN:** Designed specifically for blood clot image data, optimized for the dataset size and complexity.
- **Transfer Learning Models:**
  - **ResNet50/101:** Used for leveraging deep residual learning.
  - **VGG16/VGG19:** For extracting hierarchical image features.
  - **InceptionV3:** For capturing multi-scale features.
- All models were fine-tuned on the Mayo Clinic dataset for the specific classification task.

---

## Results

- **Accuracy:** Achieved test set accuracy between 92% and 97% depending on the model architecture.
- **Best Model:** ResNet50 fine-tuned with data augmentation achieved the highest accuracy and lowest validation loss.
- **Confusion Matrix:** Demonstrated high precision and recall across all classes, with minor confusion between visually similar classes.
- **Generalization:** The models showed strong performance on unseen data, confirming robustness.
- **Clinical Relevance:** The system demonstrates potential for real-world deployment, assisting clinicians with rapid and accurate blood clot classification.

##Model Output and Evaluation
  - **InceptionV3** and **Vgg16** performed slightly better with Vgg16 registering the lowest validation loss.
  - Evaluated model performance on test data provided by the competition:
  -   Prediction Analysis: Ensemble predictions showed improved classification probabilities for patients, increasing the confidence in correct classifications.
  -   Second approach achieved a validation accuracy of 76% and validation loss of 0.50.
---

> **Conclusion:**  
This project demonstrates the effectiveness of advanced CNN techniques in medical image classification tasks. By automating blood clot identification, it provides a valuable tool for healthcare professionals, potentially improving diagnostic workflows and patient outcomes.
