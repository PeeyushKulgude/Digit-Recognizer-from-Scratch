package com.digitrecognizer.models;

import com.digitrecognizer.utils.Value;
import java.util.ArrayList;
import java.util.List;

public class Neuron {
    Value[] weights;
    Value bias;
    int numInputs;

    public Neuron(int numInputs) {
        this.numInputs = numInputs;
        weights = new Value[numInputs];
        for (int i = 0; i < numInputs; i++) {
            weights[i] = new Value(Math.random() * 2 - 1); // Random weights between -1 and 1
        }
        bias = new Value(Math.random() * 2 - 1); // Random bias between -1 and 1
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Value weight : weights) {
            params.add(weight);
        }
        params.add(bias);
        return params;
    }

    public Value forward(Value[] inputs) {
        if (inputs.length != numInputs) {
            throw new IllegalArgumentException("Number of inputs must match number of weights");
        }

        Value sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum = sum.add(inputs[i].multiply(weights[i]));
        }
        return sum.tanh(); // Activation function
    }
}