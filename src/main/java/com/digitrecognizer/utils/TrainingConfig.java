package com.digitrecognizer.utils;

public class TrainingConfig {

    private static final String MNIST_CSV_FOLDER_PATH = "src/main/resources/mnist_export";
    private static double LEARNING_RATE = 0.06;
    private static final int BATCH_SIZE = 40;
    private static final int EPOCHS = 10;
    private static final int NUM_CLASSES = 10;
    private static final String TRAINING_OUTPUT_FILE = "training_output.txt";
    private static final String TESTING_OUTPUT_FILE = "testing_output.txt";

    public static String getMnistCsvFolderPath() {
        return MNIST_CSV_FOLDER_PATH;
    }

    public static double getLearningRate(int epoch) {
        return LEARNING_RATE;
    }

    public static void recalibrateLearningRate(int epoch) {
        LEARNING_RATE *= 0.5;
    }

    public static int getBatchSize() {
        return BATCH_SIZE;
    }

    public static int getEpochs() {
        return EPOCHS;
    }

    public static int getNumClasses() {
        return NUM_CLASSES;
    }

    public static String getTrainingOutputFile() {
        return TRAINING_OUTPUT_FILE;
    }
    
    public static String getTestingOutputFile() {
        return TESTING_OUTPUT_FILE;
    }
}
