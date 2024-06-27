from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2


def get_model(arch=2, l2_reg = 0.01, learning_rate = 0.001, optimizer = 'adam'):

    if optimizer == 'adam':
        optim = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optim = SGD(learning_rate=learning_rate)

    hidden_units = 64 if arch == 2 else 256

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(260,), activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

    return model


def train_batch(model, x, y, x_val, y_val, epochs, best_model_path):

    num_samples = x.shape[0]
    batch_size = num_samples // epochs
    best_model_acc = 0
    best_model_loss = 0

    for epoch in range(epochs):
        start_idx = epoch * batch_size
        end_idx = start_idx + batch_size
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        model.train_on_batch(x_batch, y_batch)

        if (epoch + 1) % 100 == 0:
            loss, accuracy = model.evaluate(x_batch, y_batch, verbose=0)
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            val_acc, val_loss = validate_model(
                model,
                x_val,
                y_val,
                600,
            )

            if val_acc > best_model_acc:
                best_model_acc = val_acc
                best_model_loss = val_loss
                model.save(best_model_path)
                print(
                    f"Epoch {epoch}: new best model! Accuracy: {best_model_acc:.4f} | Loss {best_model_loss:.4f}"
                )


def validate_model(model, x, y, epochs):

    import numpy as np

    num_samples = x.shape[0]
    batch_size = num_samples // epochs
    accuracies = []
    losses = []

    for epoch in range(epochs):

        start_idx = epoch * batch_size
        end_idx = start_idx + batch_size
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        loss, accuracy = model.evaluate(x_batch, y_batch, verbose=0)
        accuracies.append(accuracy)
        losses.append(loss)

    mean_acc = np.mean(accuracies)
    mean_loss = np.mean(losses)

    return mean_acc, mean_loss
