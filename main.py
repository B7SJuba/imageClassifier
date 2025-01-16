import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from skimage.feature import hog #skimage version must be 18.3 ; any version later than that could cause some probems reading rgb images!!!
import seaborn as sns
from sklearn.metrics import classification_report
import pickle


# Provide the path for training and test for each category to be classified and make sure that the data is clean,
# without any corrupted images.
SampleImage = r"path/to/a/sample/image.*"

CarTrainSet = r"path/to/car/training/images/*"
BusTrainSet = r"path/to/bus/training/images/*"
MotorTrainSet = r"path/to/motorcycle/training/images/*"
TruckTrainSet = r"path/to/truck/training/images/*"

CarTestSet = r"path/to/car/testing/images/*"
BusTestSet = r"path/to/bus/testing/images/*"
MotorTestSet = r"path/to/motorcycle/testing/images/*"
TruckTestSet = r"path/to/truck/testing/images/*"

#feature extraction technique
#first we will read the image and convert it into a numpy array
img = np.array(mpimg.imread(SampleImage))
img.setflags(write=1)
print('image: ', img.shape)
plt.imshow(img)
plt.show()

#Resizing the image to make the program runs faster
resized_img = cv2.resize(img,(64,128))
print(resized_img.shape)
plt.imshow(resized_img)
plt.show()



#Hog features gives us around 6804 features for a single image
fd, hog_image = hog(resized_img, visualize=True, multichannel=True) #Extract Histogram of Oriented Gradients (HOG) for a given image#, multichannel=True
print(fd.shape)
print(fd)
print(hog_image.shape)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()


#Training datasets
#preparing car set for training
data_car = []

for entry in glob.glob(CarTrainSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))
    fd = hog(resized_img)
    data_car.append(fd)

#preparing bus set for training
data_bus = []
for entry in glob.glob(BusTrainSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))
    fd = hog(resized_img)
    data_bus.append(fd)

#preparing motorcycle set for training
data_motorcycle = []
for entry in glob.glob(MotorTrainSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))

    fd = hog(resized_img)
    data_motorcycle.append(fd)

    # preparing truck set for training
data_truck = []
for entry in glob.glob(TruckTrainSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img, (64, 128))

    fd = hog(resized_img)
    data_truck.append(fd)

#Now we combine the training data
train_data = data_car + data_bus + data_motorcycle + data_truck
print('train data length:', len(train_data))



#Labeling
train_label_list = []
for i in range(len(train_data)):
    if i < 1158:
        train_label_list.append('car')
    elif i < 1158+1144:
        train_label_list.append('bus')
    elif i < 1158+1144+1091:
        train_label_list.append('motorcycle')
    else: #up to 1158+1144+1091+1106 images
        train_label_list.append('truck')
print(train_label_list)
print(len(train_label_list))


#Test data
#Labeled cars test data
test_car = []
for entry in glob.glob(CarTestSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))

    fd = hog(resized_img)
    test_dict = {'data':fd, 'label':'car'} #feature extraction part
    test_car.append(test_dict)

#Labeling bus test data
test_bus = []
for entry in glob.glob(BusTestSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))

    fd = hog(resized_img)
    test_dict = {'data':fd, 'label':'bus'}
    test_bus.append(test_dict)

#for motorcycle
test_motorcycle = []
for entry in glob.glob(MotorTestSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))

    fd = hog(resized_img)
    test_dict = {'data':fd, 'label':'motorcycle'}
    test_motorcycle.append(test_dict)

#for truck
test_truck = []
for entry in glob.glob(TruckTestSet):
    img = np.array(mpimg.imread(entry))
    resized_img = cv2.resize(img,(64, 128))

    fd = hog(resized_img)
    test_dict = {'data':fd, 'label':'truck'}
    test_truck.append(test_dict)

#combining the labeled test data
test_data = test_car + test_bus + test_motorcycle + test_truck
print(len(test_data))


#Now seperating the data and labels to different lists
#We will be fitting the model with the images and their labels as x and y values.
test_features = []
test_labels = []
for i in test_data:
    test_labels.append(i['label'])
    test_features.append(i['data'])
print(len(test_features))
print(test_labels)

'''
'''
#now implementing KNN
#Euclidean training and prediction with k values form 1 to 7
x_axis_k_points = []

#Metrics list
f1_euclidean = []
accuracies_euclidean = []
conf_matrix_euclidean = []


#Expereminting to figure out the best number for k (to plot remove the comments on line 234
for k in range(7):
    #knn classifier train data
    knn_euclidean = KNeighborsClassifier(n_neighbors=k+1)

    knn_euclidean.fit(train_data, train_label_list)

    #knn classifier prediction
    pred_labels_euclidean = knn_euclidean.predict(test_features) #predicting the labels depending on the features

    #accuracy of prediction
    acc_euclidean = knn_euclidean.score(test_features,test_labels)
    accuracies_euclidean.append(acc_euclidean)

    #confusion matrix of predictions
    conf_matrix_euclidean.append(metrics.confusion_matrix(test_labels,pred_labels_euclidean))

    #F1 Scores
    f1_euclidean.append(metrics.f1_score(test_labels,pred_labels_euclidean, average='weighted'))

    x_axis_k_points.append(k+1)

    print(f'When K = {k+1}')
    print(classification_report(test_labels, pred_labels_euclidean))


#knn classifier train data and saving the model
# This can be commented after the first run of the program we will be importing the saved model, but we have
# to comment the lines for metrics and scores.
knn_euclidean = KNeighborsClassifier(n_neighbors=1)

knn_euclidean.fit(train_data, train_label_list)

#Saving the trained model
with open('knn_euclidean.pickle','wb') as i:
    pickle.dump(knn_euclidean,i)

saved = open('knn_euclidean.pickle', 'rb') #loading the saved model
knn_euclidean = pickle.load(saved)


#knn classifier prediction
pred_labels_euclidean = knn_euclidean.predict(test_features) #predicting the labels depending on the features

#accuracy of prediction
acc_euclidean = knn_euclidean.score(test_features,test_labels)
accuracies_euclidean.append(acc_euclidean)

#confusion matrix of predictions
conf_matrix_euclidean.append(metrics.confusion_matrix(test_labels,pred_labels_euclidean))

#F1 Scores
f1_euclidean.append(metrics.f1_score(test_labels,pred_labels_euclidean, average='weighted'))

#x_axis_k_points.append()

print(f'When K = 1')
print(classification_report(test_labels, pred_labels_euclidean))


#showing results for euclidean
print('-----------------------------------------------------------------------------------------')

print("Euclidean scores:\n")
for i in range(len(f1_euclidean)):
      print('For k = ', i + 1, ',F1 Score= ', f1_euclidean[i], ',ACCURACY= ', accuracies_euclidean[i],
            '\nCONFUSION MATRIX:\n', conf_matrix_euclidean[i])
'''
''' #Remove comment when Experimenting to figure out the best number for k on line 166
#Plotting scores
print('-----------------------------------------------------------------------------------------')
#F1 score
plt.plot(x_axis_k_points, f1_euclidean, label='Euclidean')
plt.title("F1_Scores")
plt.xlabel("value of k")
plt.ylabel("score")
plt.legend()
plt.show()

#Accuracies
plt.plot(x_axis_k_points, accuracies_euclidean, label='Euclidean')
plt.title("Accuracies")
plt.xlabel("value of k")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
'''
'''
#confusion matrix illustration
for i in range(len(conf_matrix_euclidean)):
    s = sns.heatmap(conf_matrix_euclidean[i], annot=True, cmap='nipy_spectral_r', fmt='g')
    s.set(title=f"Confusion matrix where k = {i+1}")
    plt.show()


print('-----------------------------------------------------------------------------------------')

print('-----------------------------------------------------------------------------------------')
print('SVM')
#Now SVM module
#We start by taking the data which is ready to be fit into the model
# This can be commented after executing the first time we will be importing the saved model
from sklearn.svm import SVC #Support Vector Classifier

model_SVC = SVC(C=10, kernel='rbf')
model_SVC.fit(train_data, train_label_list) #Fitting the model

#Saving the trained model
with open('model_SVC.pickle','wb') as i:
    pickle.dump(model_SVC, i)


saved2 = open('model_SVC.pickle', 'rb') #loading the saved model
model_SVC = pickle.load(saved2)


pred_labels_SVC = model_SVC.predict(test_features)

#accuracy of prediction
acc_SVC = model_SVC.score(test_features,test_labels)

#confusion matrix of predictions
conf_matrix_SVC = metrics.confusion_matrix(test_labels,pred_labels_SVC)

#F1 Scores
f1_SVC = metrics.f1_score(test_labels,pred_labels_SVC, average='weighted')

#x_axis_k_points.append()

print("SVC Classification report:")
print(classification_report(test_labels, pred_labels_SVC))


#showing results for SVC
print('-----------------------------------------------------------------------------------------')

print('F1 Score= ', f1_SVC, ',ACCURACY= ', acc_SVC,
      '\nCONFUSION MATRIX:\n', conf_matrix_SVC)



#Plotting scores
print('-----------------------------------------------------------------------------------------')
# create a bar chart with a color gradient
fig, ax = plt.subplots()
colors = plt.cm.Reds(np.linspace(0, 1, 2))
bars = ax.bar(['Accuracy', 'F1 Score'], [acc_SVC, f1_SVC], color=colors)

#F1 score & Accuracy using bar chart because I used only one value for C, therefore a line plot would be useless
# adding labels to the bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# adding a title and axis labels
plt.title("SVC Model Performance")
plt.xlabel("Metric")
plt.ylabel("Score")

# adding a horizontal line at the mean score
mean_score = np.mean([acc_SVC, f1_SVC])
ax.axhline(mean_score, color='gray', alpha=0.5, linestyle='--')
ax.annotate(f'Mean Score: {mean_score:.2f}',
            xy=(0.5, mean_score),
            xytext=(0, 10),
            textcoords="offset points",
            ha='center', va='bottom', color='gray')

plt.show()

#confusion matrix illustration
s = sns.heatmap(conf_matrix_SVC, annot=True, cmap='nipy_spectral_r', fmt='g')
s.set(title=f"Confusion matrix")
plt.show()



def Choice():
    # You need to provide the path for the image to be classified.
    decision = input("Please Select your classification model,(1: for KNN, 2: for SVM, 0: to exit):")

    if decision == "0": print('Exiting...')

    elif decision == "1":

        print('KNN')
        print('-----------------------------------------------------------------------------------------')
        #now asking an image input from the user to classify it
        Image_path = input('Enter the Image path: ')

        input_image = []
        img = np.array(mpimg.imread(Image_path))
        resized_img = cv2.resize(img,(64, 128))

        fd = hog(resized_img)
        input_dict = {'data':fd} #feature extraction part
        input_image.append(input_dict)
        input_image = np.vstack([d['data'] for d in input_image])
        #print(input_image)
        pred_labels_euclidean = knn_euclidean.predict(input_image)


        # Use join() to convert the list to a string
        pred_labels_euclidean = ''.join(pred_labels_euclidean) #Because it's saved in a list therefore we are converting it into a single string
        print('Predicted class is:', pred_labels_euclidean)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted class is: {pred_labels_euclidean}")
        plt.show()
        print('-----------------------------------------------------------------------------------------')
        Choice() #a little bit of recursion ;D

    elif decision == "2":

        print('SVC')
        print('-----------------------------------------------------------------------------------------')
        # now asking an image input from the user to classify it with SVC

        Image_path = input('Enter the Image path: ')

        input_image = []
        img = np.array(mpimg.imread(Image_path))
        resized_img = cv2.resize(img, (64, 128))

        fd = hog(resized_img)
        input_dict = {'data': fd}  # feature extraction part
        input_image.append(input_dict)
        input_image = np.vstack([d['data'] for d in input_image])
        # print(input_image)
        pred_labels_SVC = model_SVC.predict(input_image)

        # Use join() to convert the list to a string
        pred_labels_SVC = ''.join(
            pred_labels_SVC)  # Because it's saved in a list therefore we are converting it into a single string
        print('Predicted class is:', pred_labels_SVC)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted class is: {pred_labels_SVC}")
        plt.show()
        print('-----------------------------------------------------------------------------------------')
        Choice()

#Calling the function
Choice()