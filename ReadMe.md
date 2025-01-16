# Image Classification Program

This Python program classifies images into categories such as cars, buses, motorcycles, and trucks using either the Support Vector Machine (SVM) or K-Nearest Neighbors (KNN) algorithms. It extracts image features using Histogram of Oriented Gradients (HOG) and provides an interface to classify new images interactively.

## Features

- Supports classification using **SVM** or **KNN**.
- Includes a feature extraction step using HOG.
- Saves trained models for reuse to avoid retraining.
- Provides visualizations for feature extraction and results.

## Prerequisites

### Required Libraries

Ensure you have the following libraries installed in your Python environment:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `scikit-image` (version 18.3 to avoid compatibility issues with HOG)
- `seaborn`
- `pickle`

Install the libraries using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Datasets

Provide paths for training and testing datasets for each category (car, bus, motorcycle, truck). Ensure the data is clean and free of corrupted images.

Example:

```python
CarTrainSet = r"path/to/car/training/images/*"
BusTrainSet = r"path/to/bus/training/images/*"
MotorTrainSet = r"path/to/motorcycle/training/images/*"
TruckTrainSet = r"path/to/truck/training/images/*"

CarTestSet = r"path/to/car/testing/images/*"
BusTestSet = r"path/to/bus/testing/images/*"
MotorTestSet = r"path/to/motorcycle/testing/images/*"
TruckTestSet = r"path/to/truck/testing/images/*"
```

### 2. Training the Model

The program trains both **KNN** and **SVM** models. After training, the models are saved as:

- `knn_euclidean.pickle`
- `model_SVC.pickle`

To avoid retraining, comment out the training sections in the code and load the saved models directly. If you want to train on different data, provide the appropriate paths for your datasets and run the training sections.

```python
# Uncomment to train the model
# knn_euclidean.fit(train_data, train_label_list)
# with open('knn_euclidean.pickle','wb') as i:
#     pickle.dump(knn_euclidean,i)

# Load the saved model
saved = open('knn_euclidean.pickle', 'rb')
knn_euclidean = pickle.load(saved)
```

### 3. Classifying Images

Run the program and select the classification model:

- Enter **1** for KNN.
- Enter **2** for SVM.
- Enter **0** to exit.

Provide the path to the image you want to classify when prompted. The program will display the predicted class and the image.

### 4. Visualizing Results

The program visualizes:

- Feature extraction using HOG.
- Confusion matrices.
- Accuracy and F1 score comparisons for different K values (KNN).

### Example Output

**Sample classification:**

```
Please Select your classification model,(1: for KNN, 2: for SVM, 0: to exit): 1
Enter the Image path: path/to/image.jpg
Predicted class is: car
```

## Notes

- Ensure the datasets are structured and labeled correctly.
- Experiment with the `C` parameter in SVM and the `k` parameter in KNN for optimal performance.
- HOG may fail with some newer `scikit-image` versions. Use `scikit-image==0.18.3` for compatibility.

## License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.

