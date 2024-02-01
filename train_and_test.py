import os
import pandas as pd
import numpy as np
from keras import layers, models, Model, Input
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load your CSV file
csv_file = 'train/_classes.csv' 
df = pd.read_csv(csv_file)

# Label encode the categorical columns
label_encoders = {}
for column in [' Alert', ' Smiling', ' distracted', ' eye closed', ' eye open', ' neutral']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Specify the directory path where your images are stored
image_directory = 'train'  

# Combine the directory path with the filenames
df['full_filepath'] = df['filename'].apply(lambda x: os.path.join(image_directory, x))

# Extract full image file paths and labels
image_paths = df['full_filepath'].values
labels = df[[' Alert', ' Smiling', ' distracted', ' eye closed', ' eye open', ' neutral']].values

# Split the dataset into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Create a data generator for training data with data augmentation
data_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

def data_generator_with_augmentation(image_paths, labels, batch_size=32):
    num_samples = len(image_paths)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size
            batch_paths = image_paths[indices[start:end]]
            batch_labels = labels[indices[start:end], :]

            # Load and preprocess images with data augmentation
            batch_images = [img_to_array(load_img(path, target_size=(224, 224))) for path in batch_paths]
            batch_images = [data_generator.random_transform(img) for img in batch_images]
            batch_images = [preprocess_input(img) for img in batch_images]
            batch_images = np.array(batch_images)

            yield batch_images, {
            'output_face': batch_labels[:, 0],
            'output_mouth': batch_labels[:, 1],
            'output_face': batch_labels[:, 2],
            'output_eyes': batch_labels[:, 3],
            'output_eyes': batch_labels[:, 4],
            'output_mouth': batch_labels[:, 5]
        }

# Create MobileNetV2 base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Define the input layer
inputs = Input(shape=(224, 224, 3))

# Feature extraction using the base model
x = base_model(inputs, training=False)

# Global average pooling
x = GlobalAveragePooling2D()(x)

# Common dense layers for further feature extraction
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(16, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(10, activation='relu')(x)
x = Dropout(0.5)(x)

# Output branches for face, eyes, and mouth
output_face = Dense(2, activation='softmax', name='output_face')(x)
output_eyes = Dense(2, activation='softmax', name='output_eyes')(x)
output_mouth = Dense(2, activation='softmax', name='output_mouth')(x)

# Create the model with three outputs
model = Model(inputs=inputs, outputs=[output_face, output_eyes, output_mouth])


# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),  # Adjusted learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 64
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

# Train the model with data augmentation and fine-tuning
history = model.fit(
    data_generator_with_augmentation(train_paths, train_labels, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=1,
    validation_data=data_generator_with_augmentation(val_paths, val_labels, batch_size),
    validation_steps=validation_steps
)

# Evaluate the model on a separate test set
test_csv_file = 'test/_classes.csv'
test_df = pd.read_csv(test_csv_file)
# test_df[[' Alert', ' Smiling', ' distracted', ' eye closed', ' eye open', ' neutral']] = test_df[[' Alert', ' Smiling', ' distracted', ' eye closed', ' eye open', ' neutral']].apply(lambda x: label_encoders[x.name].transform(x))
test_image_paths = test_df['filename'].apply(lambda x: os.path.join('test', x)).values
test_labels = test_df[[' Alert', ' Smiling', ' distracted', ' eye closed', ' eye open', ' neutral']].values

test_steps = len(test_image_paths) // batch_size
test_generator = data_generator_with_augmentation(test_image_paths, test_labels, batch_size)
eval_results = model.evaluate(test_generator, steps=test_steps)
print(eval_results)

# Unpack the evaluation results
test_loss_face, test_accuracy_face, test_loss_eyes, test_accuracy_eyes, test_loss_mouth, test_accuracy_mouth, _ = eval_results

# Print the results
print(f'Test Loss (Face): {test_loss_face:.4f}')
print(f'Test Accuracy (Face): {test_accuracy_face:.4f}')

print(f'Test Loss (Eyes): {test_loss_eyes:.4f}')
print(f'Test Accuracy (Eyes): {test_accuracy_eyes:.4f}')

print(f'Test Loss (Mouth): {test_loss_mouth:.4f}')
print(f'Test Accuracy (Mouth): {test_accuracy_mouth:.4f}')

# Evaluate and plot confusion matrix, F1 score, accuracy, and error for each output branch
output_columns = ['output_face', 'output_eyes', 'output_mouth']

# Evaluate the model on a separate test set
test_steps = (len(test_image_paths) + batch_size - 1) // batch_size  # Adjust test_steps for the last batch
test_generator = data_generator_with_augmentation(test_image_paths, test_labels, batch_size)
eval_results = model.evaluate(test_generator, steps=test_steps)

# Unpack the evaluation results
test_losses, *test_accuracies = eval_results

# Print the results
print(f'Test Loss: {test_losses:.4f}')

# Extract and evaluate predictions for each output branch
predictions = model.predict(test_generator, steps=test_steps)
print(predictions)
# Iterate over each output branch
for i, output_name in enumerate(output_columns):
    # Extract the predictions for the current output branch
    predicted_probabilities = predictions[i]
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    true_labels = test_labels[:, i]

    # Calculate and print F1 score
    f1 = f1_score(true_labels, predicted_labels)
    print(f'F1 Score ({output_name}): {f1:.4f}')

    # Calculate and plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({output_name})')
    plt.colorbar()

    classes = [0, 1]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Plot accuracy and error for each output branch
for i, output_name in enumerate(output_columns):
    accuracy = test_accuracies[i]
    error = 1 - accuracy

    plt.bar([f'{output_name} Accuracy', f'{output_name} Error'], [accuracy, error], color=['green', 'red'])
    plt.title(f'{output_name} Accuracy and Error')
    plt.ylabel('Percentage')
    plt.show()


# Plot accuracy and error for each output branch
for i, output_name in enumerate(output_columns):
    accuracy = test_accuracies[i]
    error = 1 - accuracy

    plt.bar([f'{output_name} Accuracy', f'{output_name} Error'], [accuracy, error], color=['green', 'red'])
    plt.title(f'{output_name} Accuracy and Error')
    plt.ylabel('Percentage')
    plt.show()


# Save the model if needed
model.save('MAIN_DMS_FinalModel.h5')