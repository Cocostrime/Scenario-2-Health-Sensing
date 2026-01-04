import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

#utiliy function
def load_dataset(dataset_dir):
    datasets = {}
    for file in os.listdir(dataset_dir):
        if file.endswith(".pkl"):
            path = os.path.join(dataset_dir, file)
            with open(path, "rb") as f:
                datasets[file.replace(".pkl","")] = pickle.load(f)
    return datasets

def prepare_data(dataset):
    """Stack flow, thorac, spo2 as 3-channel input"""
    X = np.stack([dataset['flow'], dataset['thorac'], dataset['spo2']], axis=-1)
    y = np.array(dataset['labels'])
    return X, y

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_train_cat = to_categorical(y_train_enc)
    y_test_cat = to_categorical(y_test_enc)
    return y_train_cat, y_test_cat, le

#model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_conv_lstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#evaluation matrices
def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
    y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

    acc = accuracy_score(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
    report = classification_report(y_true_labels, y_pred_labels, labels=le.classes_)
    return acc, cm, report

# ---------------- LOPO Training ----------------
def lopo_evaluation(datasets, model_type='cnn', epochs=1, batch_size=32):
    participants = list(datasets.keys())
    all_results = []

    for test_p in participants:
        print(f"\n--- LOPO: Testing on {test_p} ---")
        #spliting data into test and train
        X_test, y_test = prepare_data(datasets[test_p])
        X_train_list, y_train_list = [], []
        for train_p in participants:
            if train_p == test_p:
                continue
            X, y = prepare_data(datasets[train_p])
            X_train_list.append(X)
            y_train_list.append(y)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        #encoding labels
        y_train_cat, y_test_cat, le = encode_labels(y_train, y_test)

        #building model
        input_shape = X_train.shape[1:]  # (window_length, channels)
        num_classes = y_train_cat.shape[1]

        if model_type == 'cnn':
            model = build_cnn(input_shape, num_classes)
        else:
            model = build_conv_lstm(input_shape, num_classes)

        #training model
        model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=1)

        #evaluation of model
        acc, cm, report = evaluate_model(model, X_test, y_test_cat, le)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        all_results.append({'test_participant': test_p, 'accuracy': acc, 'confusion_matrix': cm, 'report': report})

    return all_results

#main program execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", required=True, help="Dataset folder with Pickle files")
    parser.add_argument("-model", default="cnn", choices=["cnn","conv_lstm"], help="Model type")
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=32)
    args = parser.parse_args()

    datasets = load_dataset(args.dataset_dir)
    results = lopo_evaluation(datasets, model_type=args.model, epochs=args.epochs, batch_size=args.batch_size)

    print("\n--- LOPO Evaluation Completed ---")
    for r in results:
        print(f"{r['test_participant']} Accuracy: {r['accuracy']:.4f}")