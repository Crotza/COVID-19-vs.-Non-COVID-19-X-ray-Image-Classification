
# COVID-19 vs. Non-COVID-19 X-ray Image Classification

This project focuses on the classification of X-ray images to distinguish between COVID-19 and non-COVID-19 cases. It utilizes various deep learning models for image classification and evaluates their performance.

## Project Overview

The project includes the following components:

1. Data Preparation: The X-ray image dataset is divided into training, testing, and validation sets. Image data augmentation is applied to the training set to enhance model generalization.

2. Model Architecture: Several pre-trained convolutional neural network (CNN) models, including ResNet50, VGG16, NASNetMobile, DenseNet121, MobileNet, and a custom CNN, are employed for feature extraction and classification.

3. Training: The models are trained on the training dataset with class weights applied to handle class imbalance.

4. Evaluation: The performance of each model is evaluated using metrics such as accuracy, confusion matrix, and classification report on the test dataset.

5. Visualization: The training history of each model is visualized, showing accuracy, loss, and learning rate curves.

## Dependencies

To run this project, you'll need the following Python libraries and packages:

- TensorFlow
- Matplotlib
- Numpy
- Pandas
- Seaborn
- scikit-learn

You can install these dependencies using `pip` or any other package manager.

```bash pip install tensorflow matplotlib numpy pandas seaborn scikit-learn```

## Usage

1. Clone the repository to your local machine:
```bash git clone https://github.com/your-username/covid-xray-classification.git cd covid-xray-classification```

2. Prepare your dataset:
Place your X-ray images in the dataset-comparison-2000 directory, organizing them into train, test, and validation subdirectories.

3. Run the Jupyter Notebook or Python script to train and evaluate the models:
```bash python your_script.py```

4. View the model performance metrics, training history, and classification results in the generated outputs.

## Results

The project evaluates multiple deep learning models for the classification task. Here are the accuracy results on the test dataset for each model:

- ResNet50: {95.99}%
- VGG16: {98.50}%
- NASNetMobile: {91.50}%
- DenseNet121: {98.50}%
- MobileNet: {93.00}%
- Custom CNN: {97.50}%

## Acknowledgments

- Dataset source: [https://www.kaggle.com/datasets/bachrr/covid-chest-xray]

Feel free to explore and use this project as a starting point for image classification tasks in medical diagnostics.

If you have any questions or suggestions, please contact.

