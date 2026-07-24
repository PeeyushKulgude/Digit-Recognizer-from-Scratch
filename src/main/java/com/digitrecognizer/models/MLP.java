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
            boolean isOutputLayer = (i == nouts.length - 1);
            this.layers[i] = new Layer(currentNin, nouts[i], isOutputLayer);
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

    public Value[] softmax(Value[] logits) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            if (logit.data > maxLogit) {
                maxLogit = logit.data;
            }
        }

        Value[] expValues = new Value[logits.length];
        Value sumExp = new Value(0.0);
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = logits[i].add(new Value(-maxLogit)).exp();
            sumExp = sumExp.add(expValues[i]);
        }

        Value[] softmaxOutputs = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            softmaxOutputs[i] = expValues[i].divide(sumExp);
        }

        return softmaxOutputs;
    }

    public Value crossEntropyLoss(Value[] predicted, double[] target) {
        if (predicted.length != target.length) {
            throw new IllegalArgumentException("Predicted and target lengths must match");
        }

        Value loss = new Value(0.0);
        for (int i = 0; i < predicted.length; i++) {
            loss = loss.add(new Value(-target[i]).multiply(predicted[i].log()));
        }
        return loss;
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
