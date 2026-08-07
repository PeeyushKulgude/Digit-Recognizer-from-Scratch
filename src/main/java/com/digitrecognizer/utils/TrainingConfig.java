package com.digitrecognizer.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TrainingConfig {

    private static final String MNIST_CSV_FOLDER_PATH = "src/main/resources/mnist_export";
    // Directory that all ".model" checkpoints (and their paired ".config"
    // state files) are saved to and loaded from.
    private static final String SAVED_MODELS_DIR = "saved_models";
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

    // How many epochs are still left to run out of EPOCHS. Decremented as
    // training completes epochs, saved/restored so a resumed run only trains
    // for whatever budget remains instead of starting the full EPOCHS count over.
    private static int EPOCHS_REMAINING = EPOCHS;

    public static String getMnistCsvFolderPath() {
        return MNIST_CSV_FOLDER_PATH;
    }

    public static String getSavedModelsDir() {
        return SAVED_MODELS_DIR;
    }

    /**
     * Ensures the saved-models directory exists, creating it (and any
     * missing parent directories) if necessary. Safe to call repeatedly.
     */
    public static void ensureSavedModelsDir() {
        File dir = new File(SAVED_MODELS_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
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

    /**
     * Returns just the checkpoint filename (no directory) for the given
     * epoch, e.g. "2026-07-30_10_00_epoch_4.model". Callers that need the
     * full on-disk path should join this with getSavedModelsDir().
     */
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

    public static int getEpochsRemaining() {
        return EPOCHS_REMAINING;
    }

    /**
     * Counts down the epoch budget by one. Never goes below zero.
     */
    public static void decrementEpochsRemaining() {
        if (EPOCHS_REMAINING > 0) {
            EPOCHS_REMAINING--;
        }
    }

    /**
     * Persists the mutable training config state - the learning rate (which
     * changes over time via recalibrateLearningRate) and how many epochs are
     * still left to run - so training can be resumed with the same state it
     * left off with.
     */
    public static void save(String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(Double.toString(LEARNING_RATE));
            writer.newLine();
            writer.write(Integer.toString(EPOCHS_REMAINING));
            writer.newLine();
        }
    }

    /**
     * Restores the mutable training config state previously written by save().
     * Missing or empty files/lines are treated as "nothing to restore" so
     * callers can safely call this even if no config state was ever saved.
     */
    public static void load(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String lrLine = reader.readLine();
            if (lrLine != null && !lrLine.trim().isEmpty()) {
                LEARNING_RATE = Double.parseDouble(lrLine.trim());
            }

            String remainingLine = reader.readLine();
            if (remainingLine != null && !remainingLine.trim().isEmpty()) {
                EPOCHS_REMAINING = Integer.parseInt(remainingLine.trim());
            }
        }
    }
}