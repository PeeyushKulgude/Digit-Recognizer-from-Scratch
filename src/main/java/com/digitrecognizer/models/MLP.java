package com.digitrecognizer.models;

import com.digitrecognizer.utils.Value;
import java.util.ArrayList;
import java.util.List;

public class MLP {
    private final Layer[] layers;
    public List<Value> parameters;
    private final int nin;

    public MLP(int nin, int[] nouts) {
        this.nin = nin;
        this.layers = new Layer[nouts.length];
        int currentNin = nin;
        for (int i = 0; i < nouts.length; i++) {
            this.layers[i] = new Layer(currentNin, nouts[i]);
            currentNin = nouts[i];
        }

        // Collect parameters from all layers
        this.parameters = new ArrayList<>();
        for (Layer layer : this.layers) {
            this.parameters.addAll(layer.parameters());
        }
    }

    public void zeroGrad() {
        for (Value param : this.parameters) {
            param.setGradZero();
        }
    }

    public Value[] forward(double[] inputs) {
        if (inputs.length != this.nin) {
            throw new IllegalArgumentException("Number of inputs must match the number of input features");
        }
        
        Value[] inputValues = toValueList(inputs);
        for (Layer layer : this.layers) {
            inputValues = layer.forward(inputValues);
        }
        
        return inputValues;
    }

    private Value[] toValueList(double[] input) {
        Value[] valueList = new Value[nin];
        for (int i = 0; i < nin; i++) {
            valueList[i] = new Value(input[i]);
        }
        return valueList;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("MLP Model:\n");
        for (int i = 0; i < layers.length; i++) {
            sb.append("Layer ").append(i + 1).append(": ").append(layers[i].toString()).append("\n");
        }
        return sb.toString();
    }
}
