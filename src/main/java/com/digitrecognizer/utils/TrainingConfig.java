package com.digitrecognizer.utils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

public class TrainingConfig {
    private static final Properties PROPERTIES = loadProperties();

    public static String getMnistCsvFolderPath() {
        return getString("mnistCsvFolderPath", "src/main/resources/mnist_export");
    }

    public static double getLearningRate() {
        return getDouble("learningRate", 0.01);
    }

    public static int getBatchSize() {
        return getInt("batchSize", 32);
    }

    public static int getEpochs() {
        return getInt("epochs", 10);
    }

    public static int getInputSize() {
        return getInt("inputSize", 784);
    }

    public static int getHiddenSize() {
        return getInt("hiddenSize", 128);
    }

    public static int getOutputSize() {
        return getInt("outputSize", 10);
    }

    private static Properties loadProperties() {
        Properties properties = new Properties();
        Path configPath = Paths.get("env", "training.properties");

        if (Files.exists(configPath)) {
            try (InputStream inputStream = Files.newInputStream(configPath)) {
                properties.load(inputStream);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load training properties", e);
            }
        }

        return properties;
    }

    private static String getString(String key, String defaultValue) {
        return PROPERTIES.getProperty(key, defaultValue);
    }

    private static double getDouble(String key, double defaultValue) {
        String value = PROPERTIES.getProperty(key);
        return value == null ? defaultValue : Double.parseDouble(value);
    }

    private static int getInt(String key, int defaultValue) {
        String value = PROPERTIES.getProperty(key);
        return value == null ? defaultValue : Integer.parseInt(value);
    }
}
