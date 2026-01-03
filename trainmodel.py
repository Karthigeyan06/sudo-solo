import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Data loading & preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    'D:\sudo-solo\dataset',
    target_size=(224, 224),
    batch_size=32,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'D:\sudo-solo\dataset',
    target_size=(224, 224),
    batch_size=32,
    subset='validation'
)

# 2. Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 fault categories
])

# 3. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train
model.fit(train_data, validation_data=val_data, epochs=10)

# 5. Save model
model.save('solar_fault_model.h5')
print("Model training completed and saved as solar_fault_model.h5")
