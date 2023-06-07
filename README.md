# BLINDNESS_DETECTION
 In this project, I developed a blindness detection model using machine learning techniques. Used a dataset of healthy and blind eye images, preprocessed it, and trained a convolutional neural network. Achieved high accuracy in identifying blindness, enabling early detection and intervention for improved outcomes.


BLINDNESS DETECTION.......
----->Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). At first, diabetic retinopathy may cause no symptoms or only mild vision problems.
-------->Diabetic retinopathy is caused by high blood sugar due to diabetes. Over time, having too much sugar in your blood can damage your retina — the part of your eye that detects light and sends signals to your brain through a nerve in the back of your eye (optic nerve).
-------->Diabetes damages blood vessels all over the body. The damage to your eyes starts when sugar blocks the tiny blood vessels that go to your retina, causing them to leak fluid or bleed. To make up for these blocked blood vessels, your eyes then grow new blood vessels that don’t work well. These new blood vessels can leak or bleed easily.
![download](https://github.com/06RAVI06/BLINDNESS_DETECTION/assets/107626246/68e7e624-1e50-4422-8495-453887da13d3)



-------->ABOUT THE DATA
The images consist of gaussian filtered retina scan images to detect diabetic retinopathy. The original dataset is available at APTOS 2019 Blindness Detection. These images are resized into 224x224 pixels so that they can be readily used with many pre-trained deep learning models.

-------->The project leverages two data sets:

............1. main data set used for modeling and evaluation. This data set is provided by APTOS. It has been employed in the APTOS 2019 Blindness Detection competition on Kaggle and is available for the download at the competition website: https://www.kaggle.com/c/aptos2019-blindness-detection/data. The data set includes 3,662 labeled retina images of clinical patients. The images are taken using a fundus photography technique.

.............2. supplementary data set for pre-training. This data set features 35,126 retina images labeled by a clinician using the same scale as the main data set. The data set has been used in the 2015 Diabetic Retinopathy Detection competition and is available for the download at the corresponding website: https://www.kaggle.com/c/diabetic-retinopathy-detection/data.

All of the images are already saved into their respective folders according to the severity/stage of diabetic retinopathy using the train.csv file provided. You will find five directories with the respective images:

0 - No_DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferate_DR
The dataset contains an export.pkl file which is a ResNet34 model trained on the dataset for 20 epochs using the FastAI library.
![download](https://github.com/06RAVI06/BLINDNESS_DETECTION/assets/107626246/e2b7fba1-0bac-4c64-9cd2-2d85b95aa67c)



USING CONVOLUTION NEURAL NETWORK(CNN) TO TRAIN THE MODEL
To train a blindness detection model using Convolutional Neural Networks (CNN):

1....Data Collection: Gather a dataset of eye images, including both healthy and blind cases. Ensure a diverse range of images to capture different manifestations of blindness.

2....Data Preprocessing: Clean and preprocess the collected images. This may involve resizing, normalizing, and augmenting the data to increase the size and variability of the dataset. Preprocessing helps improve the model's performance and generalization.

3.....Split the Dataset: Divide the dataset into training, validation, and testing sets. The training set is used to train the model, the validation set helps in tuning hyperparameters and monitoring performance, while the testing set evaluates the final model's accuracy.

4....Model Architecture: Design a CNN architecture suitable for the task. CNNs are well-suited for image classification tasks due to their ability to capture spatial features through convolutional layers and reduce dimensionality with pooling layers.

5....Training the Model: Train the CNN using the training dataset. This involves passing the images through the network, calculating loss, and updating the network's weights through backpropagation. Optimization algorithms like stochastic gradient descent (SGD) or Adam are commonly used during training.

6....Hyperparameter Tuning: Experiment with different hyperparameters such as learning rate, batch size, and regularization techniques to improve the model's performance. This is done by evaluating the model's performance on the validation set.

7......Model Evaluation: Once training is complete, evaluate the model's performance using the testing set. Metrics such as accuracy, precision, recall, and F1 score can be used to assess how well the model can detect blindness.

8......Deployment: After achieving satisfactory performance, the trained model can be deployed for real-world use. It can be integrated into applications or systems that assist in blindness detection, providing early intervention and improving the overall healthcare outcomes for individuals with visual impairments.

![download](https://github.com/06RAVI06/BLINDNESS_DETECTION/assets/107626246/610dcb2f-13da-4b75-af64-57cac4102f97)

