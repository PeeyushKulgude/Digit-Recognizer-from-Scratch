package com.digitrecognizer.service;

import com.digitrecognizer.utils.MnistDataLoader;
import java.util.Arrays;

public class Runner {
    private final MnistDataLoader dataLoader = new MnistDataLoader("src/main/resources/mnist_export");
    private final double[][] x_train;
    private final double[][] y_train;

    public Runner() {
        x_train = dataLoader.loadXTrain();
        System.out.println("Loaded X_train with shape: " + x_train.length + "x" + x_train[0].length);
        y_train = dataLoader.loadYTrain();
        System.out.println("Loaded y_train with shape: " + y_train.length + "x" + y_train[0].length);
        System.out.println("First 5 labels in y_train:");
        for (int i = 0; i < 5; i++) {
            System.out.println(Arrays.toString(y_train[i]));
        }
    }

    public void train() {
        
    }

    public static void main(String[] args) {
        Runner runner = new Runner();
        runner.train();
    }
}
