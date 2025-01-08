# Fundamental classes
# Time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# Image related
import seaborn as sns
import tensorflow as tf
import visualkeras
from PIL import Image
# Performance Plot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# For the model and it's training
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def date_time(x):
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:
        return 'Date today: %s' % datetime.date.today()


def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]

    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n' + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory + "/history")

    plt.show()


# Setting variables for later use
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join('images/', 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Checking data shape
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Displaying the shape after the split
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model_type = 3

if (model_type == 1):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.15))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.20))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(43, activation='softmax'))

elif (model_type == 2):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.15))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(43, activation='softmax'))

elif (model_type == 3):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(43, activation='softmax'))

else:
    print("NO MODEL SELECTED")

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model display
model.summary()

visualkeras.layered_view(model).show()
visualkeras.layered_view(model, to_file='Sieć 1/output.png')
visualkeras.layered_view(model, to_file='Sieć 1/output.png').show()

with tf.device('/GPU:0'):
    epochs = 5
    history1 = model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test, y_test))

plot_performance(history=history1)

# Predict the classes
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Print the classification report
print(classification_report(np.argmax(y_test, axis=1), y_pred_class))

# TP, FP, FN, TN
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_class)


def calculate_metrics(cm, class_idx):
    TP = cm[class_idx, class_idx]
    FP = cm[:, class_idx].sum() - TP
    FN = cm[class_idx, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    return TP, FP, FN, TN


num_classes = cm.shape[0]
metrics_matrix = []

for class_idx in range(num_classes):
    TP, FP, FN, TN = calculate_metrics(cm, class_idx)
    metrics_matrix.append([TP, FP, FN, TN])

metrics_matrix = np.array(metrics_matrix)

plt.figure(figsize=(15, 10))
sns.heatmap(
    metrics_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=["TP", "FP", "FN", "TN"], yticklabels=[f"Class {i}" for i in range(num_classes)]
)
plt.title("TP, FP, FN, TN for Each Class")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.tight_layout()
plt.show()

# Precision, Recall, and F1-Score Bar Chart
report = classification_report(np.argmax(y_test, axis=1), y_pred_class, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose().iloc[:-3]

df_report.index = df_report.index.astype(int)
df_report_sorted = df_report.sort_index()

df_report_sorted[['precision', 'recall', 'f1-score']].plot(
    kind='bar', figsize=(15, 7), legend=True
)
plt.title("Precision, Recall, and F1-Score per Class (Sorted by Class Index)")
plt.xlabel("Classes (0 to 41)")
plt.ylabel("Scores")
plt.xticks(ticks=range(len(df_report_sorted)), labels=df_report_sorted.index, rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
