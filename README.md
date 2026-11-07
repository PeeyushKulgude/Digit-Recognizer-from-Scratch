# Digit Recognizer from Scratch

A Java-based handwritten digit classifier built from the ground up with a custom neural network, manual backpropagation, and MNIST-style CSV data. The project is designed to be educational and self-contained: no external ML libraries are required, and the model can be trained, checkpointed, evaluated, and used through a simple Swing UI.

## Overview

This project implements a small multilayer perceptron (MLP) for 10-class digit recognition (0-9). It reads preprocessed MNIST data from CSV files, trains the model using cross-entropy loss, and saves checkpoints so training can resume later.

The code is split into:

- model classes for neurons, layers, and the network itself
- a runner for training/testing flows
- a data loader for MNIST CSV files
- a lightweight GUI for drawing digits and getting predictions
- configuration/state management for checkpointing and learning-rate scheduling

## Features

- Custom neural network implementation in Java
- Manual backpropagation using a small autograd-inspired `Value` class
- Softmax + cross-entropy loss
- Batch training with validation monitoring
- Learning-rate reduction on loss plateaus
- model checkpointing to `saved_models/`
- test evaluation and visual digit prediction UI

## Project Structure

```text
Digit Recognizer from Scratch/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/digitrecognizer/
│   │   │       ├── models/
│   │   │       ├── service/
│   │   │       ├── ui/
│   │   │       └── utils/
│   │   └── resources/
│   │       └── mnist_export/
│   │           ├── X_train.csv
│   │           ├── y_train.csv
│   │           ├── X_val.csv
│   │           ├── y_val.csv
│   │           ├── X_test.csv
│   │           └── y_test.csv
│   └── test/
│       └── java/
├── saved_models/
├── training_output.txt
├── testing_output.txt
├── README.md
└── src/main/java/com/digitrecognizer/utils/mnistConverter.py
```

## Prerequisites

- Java JDK 17 or newer
- A working terminal or IDE such as VS Code, IntelliJ IDEA, or Eclipse
- MNIST data exported as CSV files in `src/main/resources/mnist_export/`

## Dataset

The project expects MNIST data in the following CSV files:

- `X_train.csv` — training input images
- `y_train.csv` — one-hot or integer labels for training data
- `X_val.csv` — validation inputs
- `y_val.csv` — validation labels
- `X_test.csv` — test inputs
- `y_test.csv` — test labels

The repository includes exported CSVs in `src/main/resources/mnist_export/` and uses them directly by default.

## Getting Started

### 1. Compile the project

From the project root, compile all Java sources into the `out/` directory:

```bash
javac -d out $(find src -name "*.java")
```

If you are running on Windows PowerShell, use a similar command such as:

```powershell
Get-ChildItem -Recurse -Filter *.java -Path src | ForEach-Object { $_.FullName } | Set-Content sources.txt
javac -d out @sources.txt
```

### 2. Run the training menu

```bash
java -cp out com.digitrecognizer.service.Runner
```

The CLI offers:

1. Train a new model
2. Continue training latest model
3. Test latest model
4. Open digit-recognition GUI
5. Exit

### 3. Train from scratch

Choose option `1` to train a fresh network. The program:

- loads training and validation data
- initializes the MLP
- iterates through epochs and batches
- saves model checkpoints in `saved_models/`
- logs progress to `training_output.txt`

### 4. Continue a previous run

Choose option `2` to resume from the latest checkpoint. The code automatically looks for the newest `.model` file in `saved_models/` and restores the matching config state if available.

### 5. Test an existing model

Choose option `3` to run the latest trained model on the test set and summarize correct vs incorrect predictions.

### 6. Use the GUI

Choose option `4` to launch the Swing UI. You can draw a digit on the canvas, press Predict, and the model will return a label and confidence score.

## Model and Checkpoints

Trained models are saved in the `saved_models/` directory. Each saved model is paired with a `.config` file that stores training state such as learning rate and remaining epochs.

Example checkpoint naming:

```text
saved_models/2026-07-30_10_54_epoch_0.model
saved_models/2026-07-30_10_54_epoch_0.config
```

This allows training to resume from the most recent checkpoint without restarting from scratch.

## Training Configuration

The project configuration is centralized in `TrainingConfig.java` and includes defaults such as:

- learning rate: `0.05`
- batch size: `40`
- total epochs: `20`
- hidden layer sizes: `[128, 64, 10]`
- plateau patience: `2`
- learning-rate reduction factor: `0.5`

These values can be adjusted for experimentation.

## Test Results

Results from evaluating the checkpoint `saved_models/2026-08-10_14_16_epoch_5.model` on the 10,000-image MNIST test set (full per-sample log in `testing_output.txt`):

| Metric | Value |
| --- | --- |
| Correct predictions | 9,624 |
| Incorrect predictions | 376 |
| **Test accuracy** | **96.24%** |

Per-digit accuracy:

| Digit | Samples | Correct | Accuracy |
| --- | --- | --- | --- |
| 0 | 980 | 964 | 98.37% |
| 1 | 1,135 | 1,116 | 98.33% |
| 2 | 1,032 | 990 | 95.93% |
| 3 | 1,010 | 978 | 96.83% |
| 4 | 982 | 946 | 96.33% |
| 5 | 892 | 838 | 93.95% |
| 6 | 958 | 933 | 97.39% |
| 7 | 1,028 | 983 | 95.62% |
| 8 | 974 | 931 | 95.59% |
| 9 | 1,009 | 945 | 93.66% |

Excerpt of the test output:

```text
Loaded X_test with shape: 10000x784
Loaded y_test with shape: 10000x10
Loaded latest model from saved_models\2026-08-10_14_16_epoch_5.model to test.
Test Sample 1: Predicted Label: 7, True Label: 7
Test Sample 2: Predicted Label: 2, True Label: 2
Test Sample 3: Predicted Label: 1, True Label: 1
Test Sample 4: Predicted Label: 0, True Label: 0
Test Sample 5: Predicted Label: 4, True Label: 4
Test Sample 6: Predicted Label: 1, True Label: 1
Test Sample 7: Predicted Label: 4, True Label: 4
Test Sample 8: Predicted Label: 9, True Label: 9
Test Sample 9: Predicted Label: 6, True Label: 5
Test Sample 10: Predicted Label: 9, True Label: 9
...
Test Sample 10000: Predicted Label: 6, True Label: 6
Correct Predictions: 9624
Incorrect Predictions: 376
Testing complete.
```

## Implementation Notes

The network is intentionally small and educational. It includes:

- a custom `Value` type for scalar values and gradient tracking
- layer-wise forward propagation
- softmax and cross-entropy calculation
- gradient-based updates through parameter stepping

This is not a production-grade deep learning framework; it is a learning-focused implementation that demonstrates how a neural net can be built manually.

## Useful Files

- `src/main/java/com/digitrecognizer/service/Runner.java` — main training/testing flow
- `src/main/java/com/digitrecognizer/models/MLP.java` — network implementation
- `src/main/java/com/digitrecognizer/utils/MnistDataLoader.java` — CSV reading and label loading
- `src/main/java/com/digitrecognizer/ui/DigitRecognizerUI.java` — drawing and prediction interface
- `src/main/java/com/digitrecognizer/utils/TrainingConfig.java` — training parameters and checkpoint persistence
- `src/main/java/com/digitrecognizer/utils/mnistConverter.py` — converts pickled MNIST data to CSV export format

## Notes

- The project assumes input pixels are normalized to `[0, 1]` values (typical MNIST convention).
- If the data is exported differently, the model input scaling should be adjusted to match.
- Training performance depends on the dataset quality and the model configuration used.

## License

This project is intended for learning and experimentation. If you plan to reuse or distribute it, check your local licensing requirements and credit the original project appropriately.
