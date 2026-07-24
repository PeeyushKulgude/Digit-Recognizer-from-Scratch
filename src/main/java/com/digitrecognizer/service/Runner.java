package com.digitrecognizer.service;

import com.digitrecognizer.models.MLP;
import com.digitrecognizer.utils.MnistDataLoader;
import com.digitrecognizer.utils.TrainingConfig;
import com.digitrecognizer.utils.Value;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

public class Runner {

    private final MnistDataLoader dataLoader = new MnistDataLoader(TrainingConfig.getMnistCsvFolderPath());
    private final double[][] x_train;
    private final double[][] y_train;
    private final double[][] x_test;
    private final double[][] y_test;

    private int getTrueLabel(double[] oneHotEncoded) {
        for (int i = 0; i < oneHotEncoded.length; i++) {
            if (oneHotEncoded[i] == 1.0) {
                return i;
            }
        }
        return -1;
    }

    public Runner() {
        x_train = dataLoader.loadXTrain();
        System.out.println("Loaded X_train with shape: " + x_train.length + "x" + x_train[0].length);
        y_train = dataLoader.loadYTrain();
        System.out.println("Loaded y_train with shape: " + y_train.length + "x" + y_train[0].length);
        x_test = dataLoader.loadXTest();
        System.out.println("Loaded X_test with shape: " + x_test.length + "x" + x_test[0].length);
        y_test = dataLoader.loadYTest();
        System.out.println("Loaded y_test with shape: " + y_test.length + "x" + y_test[0].length);
    }

    public void train() {
        MLP mlp = new MLP(784, new int[]{128, 64, 10});
        int batchSize = TrainingConfig.getBatchSize();
        int batchCount = x_train.length / batchSize;

        // Training loop
        for (int epoch = 0; epoch < TrainingConfig.getEpochs(); epoch++) {
            double epochLoss = 0.0;
            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++) {
                int start = batchIndex * batchSize;
                int end = Math.min(start + batchSize, x_train.length);
                Value batchLoss = new Value(0.0);
                for (int i = start; i < end; i++) {
                    Value[] outputs = mlp.forward(x_train[i]);
                    Value[] ypred = mlp.softmax(outputs);
                    Value loss = mlp.crossEntropyLoss(ypred, y_train[i]);
                    batchLoss = batchLoss.add(loss);
                }
                batchLoss = batchLoss.multiply(new Value(1.0 / batchSize));
                epochLoss += batchLoss.data * batchSize;
                mlp.zeroGrad();
                batchLoss.backward();
                for (Value param : mlp.parameters) {
                    param.step(TrainingConfig.getLearningRate());
                }
                System.out.println("Epoch " + (epoch + 1) + ", Batch " + (batchIndex + 1) + ", Batch Loss: " + batchLoss.data);
            }
            System.out.println("Epoch " + (epoch + 1) + ", Loss: " + (epochLoss / x_train.length));
        }
        // Testing the model
        int correctPredictions = 0;
        int incorrectPredictions = 0;
        for (int i = 0; i < 100; i++) {
            int targetLabel = getTrueLabel(y_test[i]);
            Value[] outputs = mlp.forward(x_test[i]);
            Value[] ypred = mlp.softmax(outputs);
            int predictedLabel = -1;
            double maxProb = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < ypred.length; j++) {
                if (ypred[j].data > maxProb) {
                    maxProb = ypred[j].data;
                    predictedLabel = j;
                }
            }
            System.out.println("Test Sample " + (i + 1) + ": Predicted Label: " + predictedLabel + ", True Label: " + targetLabel);
            if (predictedLabel == targetLabel) {
                correctPredictions++;
            } else {
                incorrectPredictions++;
            }
        }
        System.out.println("Correct Predictions: " + correctPredictions);
        System.out.println("Incorrect Predictions: " + incorrectPredictions);
    }

    public static void main(String[] args) throws FileNotFoundException {
        File file = new File("output.txt");
        file.delete();
        PrintStream out = new PrintStream(file);
        System.setOut(out);

        Runner runner = new Runner();
        runner.train();
    }
}
