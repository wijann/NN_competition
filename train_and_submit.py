import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

#import the training dataset
df_train = pd.read_csv('train.csv')

# Drop columns not used for training
df_train.drop(['id', 'Surname', 'CustomerId'], axis=1, inplace=True)

# Use dummy encoding for categorical variables
df_train = pd.get_dummies(df_train, columns=['Geography', 'Gender'], drop_first=True)

X = df_train.drop('Exited', axis=1)
y = df_train['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y) #since my exited column is imbalanced
# Convert class weights to a dictionary
class_weights = dict(enumerate(class_weights))

def create_model(input_shape):
    model = Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification for the Zielgröße
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

model = create_model(X_scaled.shape[1])
model.fit(
    X_scaled, y,
    epochs=12, #Reducing epochs to 12 to avoid overfitting (see bank_churn.py and training validation plots)
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

df_test = pd.read_csv('test.csv')
ids = df_test['id'].copy() # Save the IDs for submission
df_test.drop(['id', 'Surname', 'CustomerId'], axis=1, inplace=True)
df_test = pd.get_dummies(df_test, columns=['Geography', 'Gender'], drop_first=True)

# Ensure the test set has the same columns as the training set
missing_cols = set(X.columns) - set(df_test.columns)
for c in missing_cols:
    df_test[c] = 0
df_test = df_test[X.columns] # reorder

X_test_scaled = scaler.transform(df_test)

preds = model.predict(X_test_scaled).flatten()
submission = pd.DataFrame({
    'id': ids, #reattach the IDs
    'Exited': preds
})

submission.to_csv('submission.csv', index=False)