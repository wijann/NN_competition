import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import the training dataset
df_train = pd.read_csv('train.csv')

# Drop columns not used for training
df_train.drop(['id', 'Surname', 'CustomerId'], axis=1, inplace=True)

# Use dummy encoding for categorical variables
df_train = pd.get_dummies(df_train, columns=['Geography', 'Gender'], drop_first=True)



# Separate input features (X) and target variable (y), keeping all data in df_train, dropping target variable for 
#k-fold cross-validation
X = df_train.drop('Exited', axis=1)
y = df_train['Exited']

# Split the data into training and testing sets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set seed for reproducibility
np.random.seed(42)

# K-Fold with 5 splits (80% train, 20% test)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results for each fold
fold_auc_scores = []
train_losses = []
val_losses = []

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

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\nTraining fold {fold + 1}")

    # Split into training and validation sets
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx] 
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define the model for each fold
    model = create_model(input_shape=X_train.shape[1])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=12, # Reduced epochs because validation loss is flattening out there, tried around with this
                        batch_size=32,
                        verbose=0, 
                        class_weight=class_weights) # Use class weights to handle class imbalance
    # Evaluate the model on the validation set
    val_auc = model.evaluate(X_val, y_val, verbose=0)[1]
    fold_auc_scores.append(val_auc)

    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

    print(f"Fold {fold + 1} AUC: {val_auc:.4f}")


print(f"\nAverage AUC across all folds: {np.mean(fold_auc_scores):.4f}")

print(f"\ntraining losses: {train_losses}")
print(f"\nvalidation losses: {val_losses}")



# AI generated code to plot the training and validation loss for each fold for simpler and quicker comparison
import matplotlib.pyplot as plt

# Convert losses to numpy arrays for easy manipulation (already done in your code)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

# Plot the training and validation loss for each fold
plt.figure(figsize=(10, 6))

# Loop through each fold's losses
for i in range(train_losses.shape[0]):  # Loop through folds
    plt.plot(train_losses[i], label=f'Train Loss - Fold {i+1}')
    plt.plot(val_losses[i], label=f'Val Loss - Fold {i+1}')

# Title and labels
plt.title('Training and Validation Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Display legend
plt.legend(loc='upper right')

# Save plot as png since it can't be displayed in the wsl environment
plt.savefig('training_validation_losses.png')
