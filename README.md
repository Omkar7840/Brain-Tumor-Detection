# Brain Tumor Detection Using CNN

This project involves detecting brain tumors from MRI images using Convolutional Neural Networks (CNNs). The model leverages transfer learning with pre-trained architectures like EfficientNetB0 and MobileNet, enhancing them for binary classification tasks.

## Project Overview

### Data Management
- The project begins by mounting Google Drive to access the dataset stored in a specific directory structure.
- The dataset is organized into two categories: 'yes' (indicating the presence of a tumor) and 'no' (indicating no tumor).
- A `dataFolder` function is defined to split the dataset into training, validation, and test sets, maintaining the integrity of the original dataset.

### Model Architecture
- **EfficientNetB0** and **MobileNet** are used as the base models.
- For both models, the top layer is removed, and custom layers are added for binary classification.
- The EfficientNetB0 model includes a Global Average Pooling layer, followed by Dropout and a Dense layer with a sigmoid activation function.
- The MobileNet model also flattens the output and adds a Dense layer with a sigmoid activation function.

### Training Process
- The models are compiled using the Adam optimizer, binary cross-entropy loss, and metrics like accuracy, precision, and recall.
- The `ImageDataGenerator` class is used for image preprocessing and augmentation.
- Callbacks such as `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` are employed to improve training efficiency and prevent overfitting.

### Evaluation
- After training, the models are evaluated on the test set to determine their performance using accuracy, precision, recall, and F1-score.
- A detailed history of the training process is plotted for both accuracy and loss.

### Prediction
- The trained model is saved and used to predict new MRI images.
- The prediction script loads the image, preprocesses it, and outputs whether the image indicates a brain tumor.

## Code Structure
- **Data Preparation**: Splits the dataset into train, validation, and test sets.
- **Model Definition**: Builds the CNN architecture using EfficientNetB0 and MobileNet.
- **Training**: Compiles and trains the model, applying data augmentation and callback functions.
- **Evaluation**: Tests the model on unseen data, calculating key metrics.
- **Prediction**: Loads a saved model to predict brain tumor presence in new MRI images.

## Dependencies
The project requires the following Python libraries:
- TensorFlow
- NumPy
- Matplotlib
- Pandas
- Seaborn
- Scikit-learn
- OpenCV

## Drive Link for Dataset
 https://drive.google.com/drive/folders/1M2W9qXIWeYvLqvOLSI2be2Yjuc99D7Ic?usp=sharing

## Image 
![Screenshot 2025-01-15 234855](https://github.com/user-attachments/assets/8cbfcf5f-ae19-4850-880c-e9835b89f397)
