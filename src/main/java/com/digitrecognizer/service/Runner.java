package com.digitrecognizer.service;

import com.digitrecognizer.utils.MnistDataLoader;
import com.digitrecognizer.utils.TrainingConfig;

public class Runner {
    private final MnistDataLoader dataLoader = new MnistDataLoader(TrainingConfig.getMnistCsvFolderPath());
    private final double[][] x_train;
    private final double[][] y_train;

    public Runner() {
        x_train = dataLoader.loadXTrain();
        System.out.println("Loaded X_train with shape: " + x_train.length + "x" + x_train[0].length);
        y_train = dataLoader.loadYTrain();
        System.out.println("Loaded y_train with shape: " + y_train.length + "x" + y_train[0].length);
        System.out.println("Using learning rate: " + TrainingConfig.getLearningRate());
        System.out.println("Using batch size: " + TrainingConfig.getBatchSize());
    }

    public void train() {
        
    }

    public static void main(String[] args) {
        Runner runner = new Runner();
        runner.train();
    }
}
