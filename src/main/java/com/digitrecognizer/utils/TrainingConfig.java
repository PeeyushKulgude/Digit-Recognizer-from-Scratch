package com.digitrecognizer.utils;

public class TrainingConfig {

    private static final String MNIST_CSV_FOLDER_PATH = "src/main/resources/mnist_export";
    private static final double LEARNING_RATE = 0.1;
    private static final int BATCH_SIZE = 40;
    private static final int EPOCHS = 1;

    public static String getMnistCsvFolderPath() {
        return MNIST_CSV_FOLDER_PATH;
    }

    public static double getLearningRate() {
        return LEARNING_RATE;
    }

    public static int getBatchSize() {
        return BATCH_SIZE;
    }

    public static int getEpochs() {
        return EPOCHS;
    }
}
