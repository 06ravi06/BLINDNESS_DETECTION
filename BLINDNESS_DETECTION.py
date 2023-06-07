----------->>>IMPORTING LIBRARIES AND MODULES
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from time import perf_counter
import seaborn as sns
​
def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))
# import system libs
import os
import time
import shutil
import pathlib
import itertools
​
# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
​
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
​
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
​
print ('modules loaded')
​
#importing libraries 
import numpy as np
import pandas as pd
​
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
​
from tqdm import tqdm_notebook as tqdm
from functools import partial
import scipy as sp
​
import random
import time
import sys
import os
​
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
​
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
import torchvision
​
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset
from torch.autograd import Variable
​
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
​
import warnings
warnings.filterwarnings('ignore')
!mkdir models

#OUTPUT
modules loaded
Requirement already satisfied: efficientnet_pytorch in c:\users\korar\ravi\lib\site-packages (0.7.1)
Requirement already satisfied: torch in c:\users\korar\ravi\lib\site-packages (from efficientnet_pytorch) (2.0.1)
Requirement already satisfied: sympy in c:\users\korar\ravi\lib\site-packages (from torch->efficientnet_pytorch) (1.10.1)
Requirement already satisfied: filelock in c:\users\korar\ravi\lib\site-packages (from torch->efficientnet_pytorch) (3.6.0)
Requirement already satisfied: networkx in c:\users\korar\ravi\lib\site-packages (from torch->efficientnet_pytorch) (2.8.4)
Requirement already satisfied: jinja2 in c:\users\korar\ravi\lib\site-packages (from torch->efficientnet_pytorch) (2.11.3)
Requirement already satisfied: typing-extensions in c:\users\korar\ravi\lib\site-packages (from torch->efficientnet_pytorch) (4.3.0)
Requirement already satisfied: MarkupSafe>=0.23 in c:\users\korar\ravi\lib\site-packages (from jinja2->torch->efficientnet_pytorch) (2.0.1)
Requirement already satisfied: mpmath>=0.19 in c:\users\korar\ravi\lib\site-packages (from sympy->torch->efficientnet_pytorch) (1.2.1)
A subdirectory or file models already exists.
-------->>>IMPORTING DATASET
image_dir = Path('C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images')
​
# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
​
# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)
​
# Shuffle the DataFrame and reset index
image_df = image_df.sample(frac=1).reset_index(drop = True)
​
# Show the result
image_df.head(25)
#OUTPUT
        Filepath	                                   Label
0	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
1	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Proliferate_DR
2	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
3	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Severe
4	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
5	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
6	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
7	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Severe
8	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Proliferate_DR
9	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
10	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
11	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
12	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
13	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
14	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
15	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
16	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
17	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
18	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
19	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Mild
20	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
21	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
22	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	Moderate
23	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
24	C:\Users\korar\OneDrive\Desktop\BLINDNESS DETE...	No_DR
--------->>>DATA PREPROCESSING
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
!pip install imutils
from imutils import paths
from sklearn.utils import shuffle
Requirement already satisfied: imutils in c:\users\korar\ravi\lib\site-packages (0.5.4)
# Set the color palette
sns.set_palette('Set2')
​
# Display some pictures of the dataset with their labels
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9),  # Increase figsize for larger images
                         subplot_kw={'xticks': [], 'yticks': []})
​
for i, ax in enumerate(axes.flat):
    image = plt.imread(image_df.Filepath[i])
    ax.imshow(image)
    ax.set_title(image_df.Label[i], color='yellow')  # Set the title color to blue
​
# Set the background color of the figure
fig.patch.set_facecolor('#333333')
​
# Set the color and size of the axis labels
for ax in axes.flat:
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    ax.xaxis.label.set_fontsize(12)  # Increase font size for axis labels
    ax.yaxis.label.set_fontsize(12)
​
# Set the color and size of the tick labels
for ax in axes.flat:
    ax.tick_params(axis='x', colors='white', labelsize=10)  # Set tick label color and size for x-axis
    ax.tick_params(axis='y', colors='white', labelsize=10)  # Set tick label color and size for y-axis
​
plt.tight_layout()
plt.show()
​
# Set a custom color palette
colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
sns.set_palette(sns.color_palette(colors))
​
# Display the number of pictures of each category
vc = image_df['Label'].value_counts()
plt.figure(figsize=(9, 5))
sns.barplot(x=vc.index, y=vc, palette=colors)  # Use the custom color palette
plt.title("Number of pictures of each category", fontsize=15)
​
# Customize the plot aesthetics
plt.xlabel("Categories", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
​
plt.show()
​

data = []
labels = []
width,height=224,224
​
imagePaths = list(paths.list_images('C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images'))
​
data = []
labels = []
​
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]   
    image = load_img(imagePath, target_size=(width, height))
    image = img_to_array(image)
    data.append(image)
    labels.append(label)
​
data = np.array(data, dtype="float32")
labels = np.array(labels)
​
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
​
data, labels = shuffle(data, labels)
​
print(data.shape)
print(labels.shape)
(3662, 224, 224, 3)
(3662, 5)
#NORMALIZING THE DATA
data = data / 255.0
---------->>>Splitting Data to Training , Validatoin and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.2)
​
print("Train images:",x_train.shape)
print("Test images:",x_test.shape)
print("Train label:",y_train.shape)
print("Test label:",y_test.shape)
Train images: (2929, 224, 224, 3)
Test images: (733, 224, 224, 3)
Train label: (2929, 5)
Test label: (733, 5)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)
​
print("Train images:",x_train.shape)
print("Test images:",x_val.shape)
print("Train label:",y_train.shape)
print("Test label:",y_val.shape)
Train images: (2343, 224, 224, 3)
Test images: (586, 224, 224, 3)
Train label: (2343, 5)
Test label: (586, 5)
import os
import pandas as pd
sdir=r'C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images'
classlist=os.listdir(sdir)    
filepaths=[]
labels=[]    
for klass in classlist:
    classpath=os.path.join(sdir,klass)
    if os.path.isdir(classpath):
        flist=os.listdir(classpath)        
        for f in flist:
            fpath=os.path.join(classpath,f)        
            filepaths.append(fpath)
            labels.append(klass)
Fseries=pd.Series(filepaths, name='filepaths')
Lseries=pd.Series(labels, name='labels')    
df=pd.concat([Fseries, Lseries], axis=1)    
print (df.head())
print('df length: ', len(df))
print (df['labels'].value_counts())
                                           filepaths                    labels
0  C:/Users/korar/OneDrive/Desktop/BLINDNESS DETE...  gaussian_filtered_images
1  C:/Users/korar/OneDrive/Desktop/BLINDNESS DETE...  gaussian_filtered_images
2  C:/Users/korar/OneDrive/Desktop/BLINDNESS DETE...  gaussian_filtered_images
3  C:/Users/korar/OneDrive/Desktop/BLINDNESS DETE...  gaussian_filtered_images
4  C:/Users/korar/OneDrive/Desktop/BLINDNESS DETE...  gaussian_filtered_images
df length:  6
gaussian_filtered_images    6
Name: labels, dtype: int64
import cv2
import matplotlib.pyplot as plt
​
# Load the image
image_path = "C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images/Mild/1d11794057ff.png"
img = cv2.imread(image_path)
​
# Check if the image is loaded successfully
if img is not None:
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
​
    # Display the image
    plt.imshow(img_rgb)
    plt.title("Mild")
    plt.show()
else:
    print("Failed to load the image.")
​

img1 = cv2.imread("C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images/Moderate/a3802934bad7.png")
plt.imshow(img1)
plt.title("Moderate")
plt.show()

img1 = cv2.imread("C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images/NO_DR/0b00f8a77510.png")
plt.imshow(img1)
plt.title("No DR")
plt.show()

img1 = cv2.imread("C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR/7efc91af4ae6.png")
plt.imshow(img1)
plt.title("PDR")
plt.show()
​

img1 = cv2.imread("C:/Users/korar/OneDrive/Desktop/BLINDNESS DETECTION/gaussian_filtered_images/gaussian_filtered_images/severe/d035c2bd9104.png")
plt.imshow(img1)
plt.title("Severe")
plt.show()

-------->>>EXAMINE MODEL ARCHITECTURE
# seed function
def seed_everything(seed = 23):
    # tests
    assert isinstance(seed, int), 'seed has to be an integer'
    
    # randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
image_size = 256
#IMAGE PREPROCESSING
​
def prepare_image(path, 
                  sigmaX         = 10, 
                  do_random_crop = False):
    
    '''
    Preprocess image
    '''
    
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)
​
    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image
def crop_black(img, tol=7):
    """
    Perform automatic crop of black areas in an image.
    """
    mask = img > tol
    
    if img.ndim == 2:
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        return img[np.ix_(mask.any(1), mask.any(0), [0, 1, 2])]
def circle_crop(img, sigmaX=10):
    """
    Perform circular crop around the image center.
    """
    height, width, _ = img.shape
​
    largest_side = max(height, width)
    img = cv2.resize(img, (largest_side, largest_side))
​
    height, width, _ = img.shape
​
    x = width // 2
    y = height // 2
    r = min(x, y)
​
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
​
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img
def random_crop(img, size=(0.9, 1)):
    """
    Random crop.
    """
​
    height, width, _ = img.shape
​
    cut = 1 - random.uniform(size[0], size[1])
​
    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)
​
    img = img[i:h, j:w, :]
    
    return img
​
​
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
​
def crop_black(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]
​
def circle_crop(img, sigmaX=10):
    height, width, _ = img.shape
    largest_side = max(height, width)
    img = cv2.resize(img, (largest_side, largest_side))
    x = width // 2
    y = height // 2
    r = min(x, y)
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img
​
class EyeData(Dataset):
    def __init__(self, data, directory, transform=None, do_random_crop=True, itype='.png'):
        self.data = data
        self.directory = directory
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
​
    def __len__(self):
        return len(self.data)
​
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'id_code'] + self.itype)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_black(image, tol=7)
        image = cv2.resize(image, (int(image_size), int(image_size)))
        image = circle_crop(image, sigmaX=10)
        image = ToTensor()(image)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': image, 'label': label}
​
class Data(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None, do_random_crop = True, itype = '.png'):
        self.data      = data
        self.directory = directory
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'id_code'] + self.itype)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_black(image, tol = 7)
        image = cv2.resize(image, (int(image_size), int(image_size)))
        image = circle_crop(image, sigmaX = 10)
        image = torch.tensor(image)
        image = image.permute(2, 1, 0)
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': image, 'label': label}
class EyeData(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None, do_random_crop = True, itype = '.png'):
        self.data      = data
        self.directory = directory
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'id_code'] + self.itype)
        image    = prepare_image(img_name, do_random_crop = self.do_random_crop)
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': image, 'label': label}
train = pd.read_csv('C:/Users/korar/OneDrive/Desktop/newwww brainnn/trainLabels.csv')
train.columns = ['id_code', 'diagnosis']
test = pd.read_csv('C:/Users/korar/OneDrive/Desktop/newwww brainnn/train.csv')
​
# check shape
print(train.shape, test.shape)
print('-' * 15)
print(train['diagnosis'].value_counts())
print('-' * 15)
print(test['diagnosis'].value_counts())
(35126, 2) (3662, 2)
---------------
0    25810
2     5292
1     2443
3      873
4      708
Name: diagnosis, dtype: int64
---------------
0    1805
2     999
1     370
4     295
3     193
Name: diagnosis, dtype: int64
fig = plt.figure(figsize = (15, 5))
plt.hist(train['diagnosis'])
plt.title('Class Distribution')
plt.ylabel('Number of examples')
plt.xlabel('Diagnosis')
Text(0.5, 0, 'Diagnosis')

----------->>>TRAINING THE DATA MODEL
def create_gen():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )
​
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
​
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
​
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
​
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images
def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False
    
    inputs = pretrained_model.input
​
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
​
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
​
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
​
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)
# Dictionary with the models
models = {
    "DenseNet121": {"model":tf.keras.applications.DenseNet121, "perf":0},
    "MobileNetV2": {"model":tf.keras.applications.MobileNetV2, "perf":0},
    "DenseNet169": {"model":tf.keras.applications.DenseNet169, "perf":0},
    "DenseNet201": {"model":tf.keras.applications.DenseNet201, "perf":0},
    "EfficientNetB0": {"model":tf.keras.applications.EfficientNetB0, "perf":0},
    "EfficientNetB1": {"model":tf.keras.applications.EfficientNetB1, "perf":0},
    "EfficientNetB2": {"model":tf.keras.applications.EfficientNetB2, "perf":0},
    "EfficientNetB3": {"model":tf.keras.applications.EfficientNetB3, "perf":0},
    "EfficientNetB4": {"model":tf.keras.applications.EfficientNetB4, "perf":0},
    "EfficientNetB5": {"model":tf.keras.applications.EfficientNetB4, "perf":0},
    "EfficientNetB6": {"model":tf.keras.applications.EfficientNetB4, "perf":0},
    "EfficientNetB7": {"model":tf.keras.applications.EfficientNetB4, "perf":0},
    "InceptionResNetV2": {"model":tf.keras.applications.InceptionResNetV2, "perf":0},
    "InceptionV3": {"model":tf.keras.applications.InceptionV3, "perf":0},
    "MobileNet": {"model":tf.keras.applications.MobileNet, "perf":0},
    "MobileNetV2": {"model":tf.keras.applications.MobileNetV2, "perf":0},
    "MobileNetV3Large": {"model":tf.keras.applications.MobileNetV3Large, "perf":0},
    "MobileNetV3Small": {"model":tf.keras.applications.MobileNetV3Small, "perf":0},
    "NASNetMobile": {"model":tf.keras.applications.NASNetMobile, "perf":0},
    "ResNet101": {"model":tf.keras.applications.ResNet101, "perf":0},
    "ResNet101V2": {"model":tf.keras.applications.ResNet101V2, "perf":0},
    "ResNet152": {"model":tf.keras.applications.ResNet152, "perf":0},
    "ResNet152V2": {"model":tf.keras.applications.ResNet152V2, "perf":0},
    "ResNet50": {"model":tf.keras.applications.ResNet50, "perf":0},
    "ResNet50V2": {"model":tf.keras.applications.ResNet50V2, "perf":0},
    "VGG16": {"model":tf.keras.applications.VGG16, "perf":0},
    "VGG19": {"model":tf.keras.applications.VGG19, "perf":0},
    "Xception": {"model":tf.keras.applications.Xception, "perf":0}
}
​
# Create the generators
train_generator,test_generator,train_images,val_images,test_images=create_gen()
print('\n')
​
# Fit the models
for name, model in models.items():
    
    # Get the model
    m = get_model(model['model'])
    models[name]['model'] = m
    
    start = perf_counter()
    
    # Fit the model
    history = m.fit(train_images,validation_data=val_images,epochs=1,verbose=1)
    
    # Sav the duration, the train_accuracy and the val_accuracy
    duration = perf_counter() - start
    duration = round(duration,2)
    models[name]['perf'] = duration
    print(f"{name:20} trained in {duration} sec")
    
    val_acc = history.history['val_accuracy']
    models[name]['val_acc'] = [round(v,4) for v in val_acc]
    
    train_acc = history.history['accuracy']
    models[name]['train_accuracy'] = [round(v,4) for v in train_acc]
Found 2966 validated image filenames belonging to 5 classes.
Found 329 validated image filenames belonging to 5 classes.
Found 367 validated image filenames belonging to 5 classes.


93/93 [==============================] - 395s 4s/step - loss: 0.8169 - accuracy: 0.7016 - val_loss: 0.7008 - val_accuracy: 0.7234
DenseNet121          trained in 396.2 sec
93/93 [==============================] - 108s 1s/step - loss: 0.8037 - accuracy: 0.7168 - val_loss: 0.6807 - val_accuracy: 0.7508
MobileNetV2          trained in 108.56 sec
93/93 [==============================] - 469s 5s/step - loss: 0.7708 - accuracy: 0.7215 - val_loss: 0.6275 - val_accuracy: 0.7599
DenseNet169          trained in 468.97 sec
93/93 [==============================] - 609s 6s/step - loss: 0.8001 - accuracy: 0.7232 - val_loss: 0.7123 - val_accuracy: 0.7295
DenseNet201          trained in 608.79 sec
93/93 [==============================] - 223s 2s/step - loss: 1.3062 - accuracy: 0.4794 - val_loss: 1.2876 - val_accuracy: 0.4924
EfficientNetB0       trained in 222.86 sec
93/93 [==============================] - 303s 3s/step - loss: 1.1868 - accuracy: 0.5175 - val_loss: 1.1230 - val_accuracy: 0.5745
EfficientNetB1       trained in 303.29 sec
93/93 [==============================] - 339s 3s/step - loss: 1.1266 - accuracy: 0.5421 - val_loss: 1.0870 - val_accuracy: 0.5653
EfficientNetB2       trained in 343.05 sec
93/93 [==============================] - 414s 4s/step - loss: 1.1074 - accuracy: 0.5823 - val_loss: 1.0295 - val_accuracy: 0.6079
EfficientNetB3       trained in 414.09 sec
93/93 [==============================] - 564s 6s/step - loss: 1.0770 - accuracy: 0.5870 - val_loss: 0.9967 - val_accuracy: 0.6201
EfficientNetB4       trained in 564.67 sec
93/93 [==============================] - 638s 7s/step - loss: 1.0668 - accuracy: 0.5954 - val_loss: 0.9908 - val_accuracy: 0.6170
EfficientNetB5       trained in 638.4 sec
93/93 [==============================] - 612s 6s/step - loss: 1.0695 - accuracy: 0.6028 - val_loss: 1.0269 - val_accuracy: 0.6170
EfficientNetB6       trained in 613.02 sec
93/93 [==============================] - 590s 6s/step - loss: 1.0565 - accuracy: 0.6015 - val_loss: 0.9914 - val_accuracy: 0.6170
EfficientNetB7       trained in 590.61 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
219055592/219055592 [==============================] - 181s 1us/step
93/93 [==============================] - 573s 6s/step - loss: 0.8508 - accuracy: 0.6969 - val_loss: 0.6843 - val_accuracy: 0.7447
InceptionResNetV2    trained in 573.38 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
87910968/87910968 [==============================] - 110s 1us/step
93/93 [==============================] - 239s 3s/step - loss: 0.8414 - accuracy: 0.7107 - val_loss: 0.7195 - val_accuracy: 0.7447
InceptionV3          trained in 238.78 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5
17225924/17225924 [==============================] - 18s 1us/step
93/93 [==============================] - 103s 1s/step - loss: 0.7813 - accuracy: 0.7208 - val_loss: 0.6502 - val_accuracy: 0.7295
MobileNet            trained in 103.29 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5
12683000/12683000 [==============================] - 17s 1us/step
93/93 [==============================] - 112s 1s/step - loss: 1.0359 - accuracy: 0.6140 - val_loss: 0.9674 - val_accuracy: 0.6383
MobileNetV3Large     trained in 112.09 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5
4334752/4334752 [==============================] - 6s 1us/step
93/93 [==============================] - 55s 547ms/step - loss: 1.1929 - accuracy: 0.5226 - val_loss: 1.1023 - val_accuracy: 0.5623
MobileNetV3Small     trained in 55.45 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-mobile-no-top.h5
19993432/19993432 [==============================] - 23s 1us/step
93/93 [==============================] - 239s 2s/step - loss: 0.8580 - accuracy: 0.7013 - val_loss: 0.7234 - val_accuracy: 0.7325
NASNetMobile         trained in 239.59 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5
171446536/171446536 [==============================] - 141s 1us/step
93/93 [==============================] - 704s 7s/step - loss: 1.0156 - accuracy: 0.6231 - val_loss: 0.9454 - val_accuracy: 0.6413
ResNet101            trained in 704.44 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5
171317808/171317808 [==============================] - 141s 1us/step
93/93 [==============================] - 638s 7s/step - loss: 0.8023 - accuracy: 0.7077 - val_loss: 0.6850 - val_accuracy: 0.7325
ResNet101V2          trained in 638.55 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5
234698864/234698864 [==============================] - 100s 0us/step
93/93 [==============================] - 1028s 11s/step - loss: 0.9699 - accuracy: 0.6450 - val_loss: 0.9072 - val_accuracy: 0.6809
ResNet152            trained in 1028.47 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5
234545216/234545216 [==============================] - 113s 0us/step
93/93 [==============================] - 928s 10s/step - loss: 0.7994 - accuracy: 0.7154 - val_loss: 0.6874 - val_accuracy: 0.7295
ResNet152V2          trained in 928.61 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94765736/94765736 [==============================] - 41s 0us/step
93/93 [==============================] - 408s 4s/step - loss: 1.0014 - accuracy: 0.6315 - val_loss: 0.9029 - val_accuracy: 0.6839
ResNet50             trained in 407.79 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
94668760/94668760 [==============================] - 41s 0us/step
93/93 [==============================] - 342s 4s/step - loss: 0.7821 - accuracy: 0.7235 - val_loss: 0.6874 - val_accuracy: 0.7416
ResNet50V2           trained in 342.05 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58889256/58889256 [==============================] - 26s 0us/step
93/93 [==============================] - 1031s 11s/step - loss: 1.0017 - accuracy: 0.6483 - val_loss: 0.8359 - val_accuracy: 0.7021
VGG16                trained in 1031.66 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
80134624/80134624 [==============================] - 35s 0us/step
93/93 [==============================] - 1274s 14s/step - loss: 0.9996 - accuracy: 0.6440 - val_loss: 0.8365 - val_accuracy: 0.7021
VGG19                trained in 1274.43 sec
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
83683744/83683744 [==============================] - 46s 1us/step
93/93 [==============================] - 424s 5s/step - loss: 0.8261 - accuracy: 0.6993 - val_loss: 0.6574 - val_accuracy: 0.7264
Xception             trained in 424.37 sec
--------->>>HYPERPARAMETER TUNING
import matplotlib.pyplot as plt
# Plot loss and kappa dynamics
plt.figure(figsize=(15, 5))
# Plot loss dynamics
plt.subplot(1, 2, 1)
plt.plot(trn_losses, color='red', label='Training')
plt.plot(val_losses, color='green', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Dynamics')
plt.legend()
​
# Plot kappa dynamics
plt.subplot(1, 2, 2)
plt.plot(val_kappas, color='blue', label='Kappa')
plt.xlabel('Epoch')
plt.ylabel('Kappa')
plt.title('Kappa Dynamics')
plt.legend()
plt.show()
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
~\AppData\Local\Temp\ipykernel_13428\1201783348.py in <module>
      4 # Plot loss dynamics
      5 plt.subplot(1, 2, 1)
----> 6 plt.plot(trn_losses, color='red', label='Training')
      7 plt.plot(val_losses, color='green', label='Validation')
      8 plt.xlabel('Epoch')

NameError: name 'trn_losses' is not defined


image.png

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
​
def init_pre_model(train=True):
    '''
    Initialize the model
    '''
    # Load pre-trained model
    if train:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=5)
    else:
        model = EfficientNet.from_name('efficientnet-b7')
        model._fc = nn.Linear(model._fc.in_features, 5)
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False
    return model
​
# Check architecture
model = init_pre_model()
print(model)
​
Loaded pretrained weights for efficientnet-b7
EfficientNet(
  (_conv_stem): Conv2dStaticSamePadding(
    3, 64, kernel_size=(3, 3), stride=(2, 2), bias=False
    (static_padding): ZeroPad2d((0, 1, 0, 1))
  )
  (_bn0): BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_blocks): ModuleList(
    (0): MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        64, 64, kernel_size=(3, 3), stride=[1, 1], groups=64, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        64, 16, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        16, 64, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (1-3): 3 x MBConvBlock(
      (_depthwise_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        32, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 32, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (4): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        192, 192, kernel_size=(3, 3), stride=[2, 2], groups=192, bias=False
        (static_padding): ZeroPad2d((0, 1, 0, 1))
      )
      (_bn1): BatchNorm2d(192, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        192, 8, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        8, 192, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (5-10): 6 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(3, 3), stride=(1, 1), groups=288, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (11): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        288, 288, kernel_size=(5, 5), stride=[2, 2], groups=288, bias=False
        (static_padding): ZeroPad2d((1, 2, 1, 2))
      )
      (_bn1): BatchNorm2d(288, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        288, 12, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        12, 288, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (12-17): 6 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(5, 5), stride=(1, 1), groups=480, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(80, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (18): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        480, 480, kernel_size=(3, 3), stride=[2, 2], groups=480, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(480, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        480, 20, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        20, 480, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (19-27): 9 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(160, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (28): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        960, 960, kernel_size=(5, 5), stride=[1, 1], groups=960, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(960, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        960, 40, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        40, 960, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (29-37): 9 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=(1, 1), groups=1344, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(224, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (38): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        1344, 1344, kernel_size=(5, 5), stride=[2, 2], groups=1344, bias=False
        (static_padding): ZeroPad2d((1, 2, 1, 2))
      )
      (_bn1): BatchNorm2d(1344, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        1344, 56, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        56, 1344, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (39-50): 12 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(5, 5), stride=(1, 1), groups=2304, bias=False
        (static_padding): ZeroPad2d((2, 2, 2, 2))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(384, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (51): MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        2304, 2304, kernel_size=(3, 3), stride=[1, 1], groups=2304, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(2304, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        2304, 96, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        96, 2304, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
    (52-54): 3 x MBConvBlock(
      (_expand_conv): Conv2dStaticSamePadding(
        640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn0): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_depthwise_conv): Conv2dStaticSamePadding(
        3840, 3840, kernel_size=(3, 3), stride=(1, 1), groups=3840, bias=False
        (static_padding): ZeroPad2d((1, 1, 1, 1))
      )
      (_bn1): BatchNorm2d(3840, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_se_reduce): Conv2dStaticSamePadding(
        3840, 160, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_se_expand): Conv2dStaticSamePadding(
        160, 3840, kernel_size=(1, 1), stride=(1, 1)
        (static_padding): Identity()
      )
      (_project_conv): Conv2dStaticSamePadding(
        3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False
        (static_padding): Identity()
      )
      (_bn2): BatchNorm2d(640, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
      (_swish): MemoryEfficientSwish()
    )
  )
  (_conv_head): Conv2dStaticSamePadding(
    640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False
    (static_padding): Identity()
  )
  (_bn1): BatchNorm2d(2560, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_avg_pooling): AdaptiveAvgPool2d(output_size=1)
  (_dropout): Dropout(p=0.5, inplace=False)
  (_fc): Linear(in_features=2560, out_features=5, bias=True)
  (_swish): MemoryEfficientSwish()
)
-------->>> EVALUATING THE TRAINED MODEL
models_result = []
​
for name, v in models.items():
    models_result.append([ name, 
                          models[name]['train_accuracy'][-1],
                          models[name]['val_acc'][-1], 
                          models[name]['perf']])
    
df_results = pd.DataFrame(models_result, 
                          columns = ['model','train_accuracy','val_accuracy','Training time (sec)'])
df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
df_results.reset_index(inplace=True,drop=True)
df_results
​
model	train_accuracy	val_accuracy	Training time (sec)
0	DenseNet169	0.7215	0.7599	468.97
1	MobileNetV2	0.7168	0.7508	108.56
2	InceptionV3	0.7107	0.7447	238.78
3	InceptionResNetV2	0.6969	0.7447	573.38
4	ResNet50V2	0.7235	0.7416	342.05
5	ResNet101V2	0.7077	0.7325	638.55
6	NASNetMobile	0.7013	0.7325	239.59
7	DenseNet201	0.7232	0.7295	608.79
8	ResNet152V2	0.7154	0.7295	928.61
9	MobileNet	0.7208	0.7295	103.29
10	Xception	0.6993	0.7264	424.37
11	DenseNet121	0.7016	0.7234	396.20
12	VGG16	0.6483	0.7021	1031.66
13	VGG19	0.6440	0.7021	1274.43
14	ResNet50	0.6315	0.6839	407.79
15	ResNet152	0.6450	0.6809	1028.47
16	ResNet101	0.6231	0.6413	704.44
17	MobileNetV3Large	0.6140	0.6383	112.09
18	EfficientNetB4	0.5870	0.6201	564.67
19	EfficientNetB7	0.6015	0.6170	590.61
20	EfficientNetB6	0.6028	0.6170	613.02
21	EfficientNetB5	0.5954	0.6170	638.40
22	EfficientNetB3	0.5823	0.6079	414.09
23	EfficientNetB1	0.5175	0.5745	303.29
24	EfficientNetB2	0.5421	0.5653	343.05
25	MobileNetV3Small	0.5226	0.5623	55.45
26	EfficientNetB0	0.4794	0.4924	222.86
plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'train_accuracy', data = df_results)
plt.title('Accuracy on the Training Set (after 1 epoch)', fontsize = 15)
plt.ylim(0,1)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'val_accuracy', data = df_results)
plt.title('Accuracy on the Validation Set (after 1 epoch)', fontsize = 15)
plt.ylim(0,1)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'Training time (sec)', data = df_results)
plt.title('Training time for each model in sec', fontsize = 15)
plt.xticks(rotation=90)
plt.show()

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
​
pretrained_model.trainable = False
import tensorflow as tf
​
# Assuming you have train_images and val_images as your actual data
​
inputs = pretrained_model.input
​
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
​
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
​
model = tf.keras.Model(inputs=inputs, outputs=outputs)
​
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
​
history = model.fit(
    train_images,  # Replace with your actual training image data
    validation_data=val_images,  # Replace with your actual validation image data
    batch_size=32,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)
​
Epoch 1/10
93/93 [==============================] - 110s 1s/step - loss: 0.8111 - accuracy: 0.7138 - val_loss: 0.6957 - val_accuracy: 0.7508
Epoch 2/10
93/93 [==============================] - 103s 1s/step - loss: 0.6397 - accuracy: 0.7680 - val_loss: 0.6729 - val_accuracy: 0.7386
Epoch 3/10
93/93 [==============================] - 110s 1s/step - loss: 0.5934 - accuracy: 0.7815 - val_loss: 0.6479 - val_accuracy: 0.7447
Epoch 4/10
93/93 [==============================] - 102s 1s/step - loss: 0.5477 - accuracy: 0.7953 - val_loss: 0.6381 - val_accuracy: 0.7508
Epoch 5/10
93/93 [==============================] - 101s 1s/step - loss: 0.5339 - accuracy: 0.7970 - val_loss: 0.6155 - val_accuracy: 0.7599
Epoch 6/10
93/93 [==============================] - 102s 1s/step - loss: 0.4937 - accuracy: 0.8068 - val_loss: 0.6589 - val_accuracy: 0.7447
Epoch 7/10
93/93 [==============================] - 102s 1s/step - loss: 0.4562 - accuracy: 0.8291 - val_loss: 0.6788 - val_accuracy: 0.7204
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()

results = model.evaluate(test_images, verbose=0)
printmd(" ## Test Loss: {:.5f}".format(results[0]))
printmd("## Accuracy on the test set: {:.2f}%".format(results[1] * 100))
Test Loss: 0.55842
Accuracy on the test set: 78.75%
import numpy as np
​
# Assuming you have test_images as your actual test image data
​
results = model.evaluate(test_images, verbose=0)
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)
​
# Map the labels
labels = train_images.class_indices
labels = {v: k for k, v in labels.items()}
pred = [labels[k] for k in pred]
​
# Display the result
print(f'The first 5 predictions: {pred[:5]}')
​
12/12 [==============================] - 24s 1s/step
The first 5 predictions: ['No_DR', 'No_DR', 'Severe', 'No_DR', 'No_DR']
from sklearn.metrics import classification_report
​
# Assuming you have assigned a value to the pred variable
​
y_test = list(test_df.Label)
print(classification_report(y_test, pred))
​
                precision    recall  f1-score   support

          Mild       0.64      0.40      0.49        40
      Moderate       0.67      0.83      0.74        93
         No_DR       0.94      0.98      0.96       194
Proliferate_DR       0.20      0.04      0.06        26
        Severe       0.25      0.36      0.29        14

      accuracy                           0.79       367
     macro avg       0.54      0.52      0.51       367
  weighted avg       0.76      0.79      0.76       367

from sklearn.metrics import confusion_matrix
import seaborn as sns
​
cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)))
plt.title('Normalized Confusion Matrix')
plt.show()
​

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})
​
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred[i]}")
plt.tight_layout()
plt.show()

-------->>>DEPLOYEMENT
import tensorflow as tf
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size "size"
    array = np.expand_dims(array, axis=0)
    return array
​
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
​
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
​
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
​
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
​
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
​
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
​
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
​
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
​
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
​
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
​
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
​
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
​
    # Save the superimposed image
    superimposed_img.save(cam_path)
​
    # Display Grad CAM
#     display(Image(cam_path))
    
    return cam_path
    
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
​
last_conv_layer_name = "Conv_1"
img_size = (224,224)
​
# Remove last layer's softmax
model.layers[-1].activation = None
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 10),
                        subplot_kw={'xticks': [], 'yticks': []})
​
for i, ax in enumerate(axes.flat):
    img_path = test_df.Filepath.iloc[i]
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    cam_path = save_and_display_gradcam(img_path, heatmap)
    ax.imshow(plt.imread(cam_path))
    ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred[i]}")
plt.tight_layout()
plt.show()

---------------------------------------------------------------------------------------------------------------¶
