# Brain_Tumor_Classification
My study presents an innovative approach to develop a robust and accurate neural network model, leveraging transfer learning with EfficientNet architecture, for the classification of brain tumors solely based on MRI scans. 
import os
import cv2
import glob
import random
import imageio
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import skimage.measure   
import tensorflow as tf
from skimage import data
import albumentations as A
import scipy.ndimage as ndi
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from tensorflow.keras import regularizers
from skimage.measure.entropy import shannon_entropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam , Adamax
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix , classification_report
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,BatchNormalization

# Training data paths
train_glioma = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Training\\glioma_tumor\\*.jpg')
train_menignioma = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Training\\meningioma_tumor\\*.jpg')
train_pituitary = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Training\\pituitary_tumor\\*.jpg')
train_no = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Training\\no_tumor\\*.jpg')


# Testing data paths
test_glioma = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Testing\\glioma_tumor\\*.jpg')
test_menignioma = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Testing\\meningioma_tumor\\*.jpg')
test_pituitary = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Testing\\pituitary_tumor\\*.jpg')
test_no = glob.glob('D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Testing\\no_tumor\\*.jpg')


train_glioma_number = len(train_glioma)
train_menignioma_number = len(train_menignioma)
train_pituitary_number = len(train_pituitary)
train_no_number = len(train_no)

test_glioma_number = len(test_glioma)
test_menignioma_number = len(test_menignioma)
test_pituitary_number = len(test_pituitary)
test_no_number = len(test_no)

print("Number of train_glioma: ",train_glioma_number)
print("Number of train_menignioma: ",train_menignioma_number)
print("Number of train_pituitary: ",train_pituitary_number)
print("Number of train_no: ",train_no_number)
print("Number of test_glioma: ",test_glioma_number)
print("Number of test_menignioma: ",train_menignioma_number)
print("Number of test_pituitary: ",train_pituitary_number)
print("Number of test_no: ",train_no_number)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/28164f4f-1d0f-484a-87a6-cf37d957c9c3)

plt.figure(figsize=(5,5))
colors = ['#4285f4','#ea4335',"#e94cdc","#00FFFF"]
plt.rcParams.update({'font.size': 8})
plt.pie([train_glioma_number,
         train_menignioma_number,train_pituitary_number,train_no_number],
        labels=['Glioma', 'Menignioma','Pituitary','No_tumor'],
        colors=colors, autopct='%.1f%%', explode=(0.025,0.025,0.025,0.025),
        startangle=30);
plt.show()
plt.figure(figsize=(5,5))
colors = ["#e94cdc","#00FFFF"]
plt.rcParams.update({'font.size': 8})
plt.pie([test_glioma_number+
         test_menignioma_number+test_pituitary_number+test_no_number,train_glioma_number+
         train_menignioma_number+train_pituitary_number+train_no_number],
        labels=['test', 'train'],
        colors=colors, autopct='%.1f%%', explode=(0.025,0.025),
        startangle=30);

print('total train data:',(train_glioma_number+
         train_menignioma_number+train_pituitary_number+train_no_number))
print('total test data:',(test_glioma_number+test_menignioma_number+test_pituitary_number+test_no_number))

#Image Visualization
def images_visualization(image, title, a):
    plt.figure(figsize=(8, 8))  # Create a single figure for all images
    for i in range(a):
        random_image_path = random.choice(image)  # Randomly select an image
        img = mpimg.imread(random_image_path)
        plt.subplot(1, a, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(title)
    
    plt.show()  # Display the figure with all images
    plt.close() 


print("Train image visualisation\n")
images_visualization(train_glioma, 'Glioma',3)
images_visualization(train_menignioma, 'Menignioma',3)
images_visualization(train_pituitary, 'Pituitary',3)
images_visualization(train_no, 'No-Tumor',3)

![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/998a7c0b-4ac9-4c31-8875-cc49beab752c)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/7e2cbdc5-e437-4250-a093-3bf5006a0906)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/6b8c2439-85a7-44fa-a1d0-c0c783012de2)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/29756c38-f9cd-4fcf-9983-ee29b733fa54)

#Convolutoin Filter definition
def fiter(image, title, a, m):
    plt.figure(figsize=(8, 8))  # Create a single figure for all images
    for i in range(a):
        random_image_path = random.choice(image)  # Randomly select an image
        img = cv2.imread(random_image_path)  # Read the image using cv2
        laplacian_kernel = np.array(m)  # Define the Laplacian kernel
        laplacian_img = cv2.filter2D(img, -1, laplacian_kernel)  # Apply the Laplacian filter using cv2.filter2D
        plt.subplot(1, a, i + 1)
        plt.imshow(laplacian_img, cmap='jet')  # Display the filtered image using plt.imshow and cmap='jet'
        plt.axis('off')
        plt.title(title)
    plt.show()
    plt.close()



m = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]# Gaussian filter 
print("Train image after gaussian filter visualisation\n")
fiter(train_glioma, 'Glioma',3,m)
fiter(train_menignioma, 'Menignioma',3,m)
fiter(train_pituitary, 'Pituitary',3,m)
fiter(train_no, 'No-Tumor',3,m)






![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/69f536e5-6c4a-4916-84a0-c6ea40a8096f)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/a2cb0e8b-d0a5-48f0-ace6-08caa88b980b)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/43f50e9d-bf24-42b6-8d46-19a0c33050fc)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/5f31beb3-1a8f-4abd-8c88-f7fd044e264e)


m =  [[1, 1, 1], [1, -8, 1], [1, 1, 1]]#laplacian of Gaussian filter 
print("Train image after laplacian of Gaussian filter1 visualisation\n")
fiter(train_glioma, 'Glioma',3,m)
fiter(train_menignioma, 'Menignioma',3,m)
fiter(train_pituitary, 'Pituitary',3,m)
fiter(train_no, 'No-Tumor',3,m)




![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/77640d3f-44b0-466c-9caa-645c778c04af)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/989a50b5-1171-430a-a38c-33c0fd8f3e94)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/fb7a2492-45f4-4025-b951-6a01db265f55)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/6b407ee4-4e65-4457-9ddb-890dd43517f3)


m =  [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]#laplacian of Gaussian filter 
print("Train image after laplacian of Gaussian filter2 visualisation\n")
fiter(train_glioma, 'Glioma',3,m)
fiter(train_menignioma, 'Menignioma',3,m)
fiter(train_pituitary, 'Pituitary',3,m)
fiter(train_no, 'No-Tumor',3,m)





![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/4b59db4b-565f-4d1c-854a-23c6fdf05a5f)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/aeda5921-17b8-4147-8cdc-91ce83d35b41)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/ccab5850-8a23-4e41-aed3-fbd4ecd90a93)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/e7a74705-e100-41eb-8930-35859282e0b9)

#clahe Filter function definition
def clahe(image, title, a):
    plt.figure(figsize=(8, 8))  # Create a single figure for all images
    for i in range(a):
        random_image_path = random.choice(image)  # Randomly select an image
        img = cv2.imread(random_image_path)  # Read the image using cv2
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))  # Create a CLAHE object with clip limit and tile grid size
        clahe_img = clahe.apply(img_gray)  # Apply CLAHE to the grayscale image
        plt.subplot(1, a, i + 1)
        plt.imshow(clahe_img, cmap='jet')  # Display the filtered image using plt.imshow and cmap='jet'
        plt.axis('off')
        plt.title(title)
 
print("Train image after clahe Filter visualisation\n")
clahe(train_glioma, 'Glioma',3)
clahe(train_menignioma, 'Menignioma',3)
clahe(train_pituitary, 'Pituitary',3)
clahe(train_no,'No-Tumor',3)
plt.show()




![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/6d8a8f4e-1bba-42e5-b8c0-0b46241e0d3a)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/aed2a175-d89e-478b-a2a6-f6e9ad479a29)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/5296c533-3977-4a72-81db-192a78114c6b)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/ae78bab5-a4a8-4cb4-8131-e1bffeb7ad17)


#Entropy Shanon filter function definition
def entr(image, title, a):
    plt.figure(figsize=(8, 8))  # Create a single figure for all images
    for i in range(a):
        random_image_path = random.choice(image)  # Randomly select an image
        img = cv2.imread(random_image_path)  # Read the image using cv2
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        entr_img = entropy(img_gray, disk(2))  # Calculate entropy of the grayscale image
        plt.subplot(1, a, i + 1)
        plt.imshow(entr_img, cmap='viridis')  # Display the entropy image using plt.imshow and cmap='viridis'
        plt.axis('off')
        plt.title(title)
#Entropy shanon Filter
print("Train image after Entropy shanon Filter visualisation\n")
entr(train_glioma, 'Glioma',3)
entr(train_menignioma, 'Menignioma',3)
entr(train_pituitary, 'Pituitary',3)
entr(train_no,'No-Tumor',3)
plt.show()




![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/05ebe1f4-c2af-4bbb-b973-06d0abfeee81)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/3f1724ce-4a60-46ca-8ea4-2e9a90da6755)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/24fe557a-1e9b-4bf3-b1b9-1912b62db5bc)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/49b4201c-e441-4ead-8a87-1e1f89e4bbe8)

#model creation is image processing
#Image dataset creation
train_data_path = 'D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Training'

filepaths =[]
labels = []

folds = os.listdir(train_data_path)

for fold in folds:
    f_path = os.path.join(train_data_path , fold)
    filelists = os.listdir(f_path)
    
    for file in filelists:
        filepaths.append(os.path.join(f_path , file))
        labels.append(fold)
        
#Concat data paths with labels
Fseries = pd.Series(filepaths , name = 'filepaths')
Lseries = pd.Series(labels , name = 'label')
train_df = pd.concat([Fseries , Lseries] , axis = 1)

train_df
print("trainnd data info\n")
print(train_df)


![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/873b8660-31de-4c21-8794-9770e6669f87)

![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/48cb43ed-3ba1-4903-a9ab-f0037443d539)

#test_data
test_data_path = 'D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\Testing'

filepaths =[]
labels = []

folds = os.listdir(test_data_path)

for fold in folds:
    f_path = os.path.join(test_data_path , fold)
    filelists = os.listdir(f_path)
    
    for file in filelists:
        filepaths.append(os.path.join(f_path , file))
        labels.append(fold)
        
#Concat data paths with labels
Fseries = pd.Series(filepaths , name = 'filepaths')
Lseries = pd.Series(labels , name = 'label')
test_df = pd.concat([Fseries , Lseries] , axis = 1)

print("test data info\n")
print(test_df)

![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/48cb43ed-3ba1-4903-a9ab-f0037443d539)
#Split dataset¶
valid , test = train_test_split(test_df , train_size = 0.5 , shuffle = True , random_state= 42)

#Image resize and generation
img_size = (220 ,220)#This defines a variable called img_size that stores a tuple of two integers, 220 and 220. This will be used as the target size for resizing the images.
batch_size = 15 # This defines a variable called batch_size that stores an integer, 16. This will be used as the number of images per batch.

tr_gen = ImageDataGenerator()#This creates an instance of the ImageDataGenerator class and assigns it to a variable called tr_gen. This will be used to create the train_gen object later. By default, this does not apply any preprocessing or augmentation to the images, but you can pass some arguments to the constructor to customize it.
ts_gen= ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df , x_col = 'filepaths' , y_col = 'label' , target_size = img_size ,
                                      class_mode = 'categorical' , color_mode = 'rgb' , shuffle = True , batch_size =batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid , x_col = 'filepaths' , y_col = 'label' , target_size = img_size , 
                                       class_mode = 'categorical',color_mode = 'rgb' , shuffle= True, batch_size = batch_size)

test_gen = ts_gen.flow_from_dataframe(test , x_col= 'filepaths' , y_col = 'label' , target_size = img_size , 
                                      class_mode = 'categorical' , color_mode= 'rgb' , shuffle = False , batch_size = batch_size)


![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/739b3917-3d8d-4037-a85c-8216d082d3a0)

#model architecture
gen_dict = train_gen.class_indices
classes = list(gen_dict.keys())
images , labels = next(train_gen)
img_shape = (img_size[0] , img_size[1] , 3)
num_class = len(classes)

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False , weights = 'imagenet' ,

                                                               input_shape = img_shape, pooling= 'max')
model = Sequential([
    base_model,
    BatchNormalization(axis= -1 , momentum= 0.99 , epsilon= 0.001),
    Dense(256, kernel_regularizer = regularizers.l2(l= 0.016) , activity_regularizer = regularizers.l1(0.006),
         bias_regularizer= regularizers.l1(0.006) , activation = 'relu'),
    Dropout(rate= 0.4 , seed = 75),
    Dense(num_class , activation = 'softmax')
])
#optimizer
model.compile(Adamax(learning_rate = 0.001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/c48a0272-c125-4547-9788-1ae4e32c9a70)


#Model training
Epochs = 8

history = model.fit(x= train_gen , epochs = Epochs , verbose = 1 , validation_data = valid_gen ,
                   validation_steps = None , shuffle = False)


model.save("brain_tumor_classification_model.h5")


# Load the trained model
model = load_model("brain_tumor_classification_model.h5")

# Define the image size
img_size = (220, 220)
new_image_path = 'D:\\Final research\\Brain-Tumor-Classification-DataSet-master\\demo\\gg (18).jpg'  # Updated file path

# Load and preprocess the new image using EfficientNet's preprocessing
new_img = image.load_img(new_image_path, target_size=img_size)
new_img = image.img_to_array(new_img)
new_img = np.expand_dims(new_img, axis=0)
new_img = preprocess_input(new_img)  # No need to specify 'mode'

# Use the trained model to make predictions
predictions = model.predict(new_img)

# Interpret the model's predictions
predicted_class = np.argmax(predictions)
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Define your class labels
predicted_label = class_labels[predicted_class]

print(f"Predicted Tumor Type: {predicted_label}")

img = mpimg.imread(new_image_path)
plt.imshow(img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/70baf985-7ce9-48b2-ae1c-0641d85d2191)
![image](https://github.com/Hauntedwolf2000/Brain_Tumor_Classification/assets/69667787/dedf8383-0178-4741-9aa4-27e6031cf9b7)
