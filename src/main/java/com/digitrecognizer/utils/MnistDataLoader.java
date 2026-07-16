package com.digitrecognizer.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class MnistDataLoader {
    private final String mnistCsvFolderPath;

    public MnistDataLoader(String mnistCsvFolderPath) {
        this.mnistCsvFolderPath = mnistCsvFolderPath;
    }

    public double[][] loadXTrain() {
        return loadMatrix("X_train.csv");
    }

    public double[][] loadYTrain() {
        return loadY("y_train.csv");
    }

    public double[][] loadXTest() {
        return loadMatrix("X_test.csv");
    }

    public double[][] loadYTest() {
        return loadY("y_test.csv");
    }

    public double[][] loadXValidation() {
        return loadMatrix("X_val.csv");
    }

    public double[][] loadYValidation() {
        return loadY("y_val.csv");
    }

    private double[][] loadMatrix(String fileName) {
        List<double[]> rows = new ArrayList<>();
        for (String line : readLines(fileName)) {
            if (line == null || line.trim().isEmpty()) {
                continue;
            }
            String[] values = line.split(",");
            double[] row = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                row[i] = Double.parseDouble(values[i]);
            }
            rows.add(row);
        }
        return rows.toArray(new double[0][0]);
    }

    private double[][] loadY(String fileName) {
        List<double[]> rows = new ArrayList<>();
        for (String line : readLines(fileName)) {
            if (line == null || line.trim().isEmpty()) {
                continue;
            }
            int label = Integer.parseInt(line.trim());
            rows.add(oneHotEncode(label, 10));
        }
        return rows.toArray(new double[0][0]);
    }

    private double[] oneHotEncode(int label, int numClasses) {
        double[] encoded = new double[numClasses];
        encoded[label] = 1.0;
        return encoded;
    }

    private List<String> readLines(String fileName) {

        String filePath = mnistCsvFolderPath + "/" + fileName;

        try {
            return java.nio.file.Files.readAllLines(
                    java.nio.file.Paths.get(filePath),
                    StandardCharsets.UTF_8
            );
        } catch (IOException e) {
            throw new RuntimeException(
                    "Failed to read file: " + filePath,
                    e
            );
        }
    }
}