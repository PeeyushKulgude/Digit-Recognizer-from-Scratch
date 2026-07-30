package com.digitrecognizer.utils;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TrainingConfig {

    private static final String MNIST_CSV_FOLDER_PATH = "src/main/resources/mnist_export";
    private static double LEARNING_RATE = 0.05;
    private static final int BATCH_SIZE = 40;
    private static final int EPOCHS = 20;
    private static final int NUM_CLASSES = 10;
    private static final String TRAINING_OUTPUT_FILE = "training_output.txt";
    private static final String TESTING_OUTPUT_FILE = "testing_output.txt";

    // Plateau detection: how many epochs with no meaningful validation-loss
    // improvement before we react (reduce LR, or stop if LR is already exhausted).
    private static final int PLATEAU_PATIENCE = 2;
    // Minimum drop in validation loss to count as "improvement".
    private static final double MIN_DELTA = 1e-3;
    // How many times we're willing to halve the LR before giving up and stopping.
    private static final int MAX_LEARNING_RATE_REDUCTIONS = 3;

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

    public static String getCurrentModelName(int epoch) {
        return DateTimeFormatter.ofPattern("yyyy-MM-dd_HH_mm").format(LocalDateTime.now()) + "_epoch_" + epoch + ".model";
    }

    public static int getPlateauPatience() {
        return PLATEAU_PATIENCE;
    }

    public static double getMinDelta() {
        return MIN_DELTA;
    }

    public static int getMaxLearningRateReductions() {
        return MAX_LEARNING_RATE_REDUCTIONS;
    }
}