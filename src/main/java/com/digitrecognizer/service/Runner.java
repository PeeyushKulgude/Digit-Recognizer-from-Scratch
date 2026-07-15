package com.digitrecognizer.service;

import com.digitrecognizer.models.MLP;
import com.digitrecognizer.utils.Value;

public class Runner {
    double[][] xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    double[] ys = {1.0, -1.0, -1.0, 1.0};

    public void run() {
        MLP mlp = new MLP(3, new int[]{4, 4, 1});
        System.out.println(mlp);
        
        for (int epoch = 0; epoch < 100000; epoch++) {

            //Forward passes
            Value totalLoss = new Value(0);
            double[] ypred = new double[xs.length];
            for (int i = 0; i < xs.length; i++) {
                double[] x = xs[i];
                double y = ys[i];

                Value output = mlp.forward(x)[0];

                // Saving output
                ypred[i] = output.data;

                Value loss = output.add(new Value(-y)).multiply(output.add(new Value(-y))); // (output - y)^2
                totalLoss = totalLoss.add(loss);
            }

            // Zero gradients before backward pass
            mlp.zeroGrad();

            // Backward pass and parameter update would go here
            totalLoss.backward();

            // Update parameters (simple gradient descent)
            for (Value param : mlp.parameters) {
                param.step(0.01); // Learning rate of 0.01
            }

            System.out.println("Epoch " + (epoch + 1) + ": Loss = " + totalLoss + ", Predictions = " + java.util.Arrays.toString(ypred));
        }

        System.out.print(mlp);
    }

    public static void main(String[] args) {
        Runner runner = new Runner();
        runner.run();
    }
}
