package com.digitrecognizer.models;

import com.digitrecognizer.utils.Value;
import java.util.ArrayList;
import java.util.List;

public class Layer {

    private final Neuron[] neurons;
    private final int nin, nout;
    private final boolean isOutputLayer;

    public Layer(int nin, int nout, boolean isOutputLayer) {
        this.nin = nin;
        this.nout = nout;
        this.isOutputLayer = isOutputLayer;

        this.neurons = new Neuron[this.nout];
        for (int i = 0; i < this.nout; i++) {
            this.neurons[i] = new Neuron(this.nin, this.isOutputLayer);
        }
    }

    public Value[] forward(Value[] inputs) {
        if (inputs.length != this.nin) {
            throw new IllegalArgumentException("Number of inputs must match number of neurons in the layer");
        }

        Value[] outputs = new Value[this.nout];
        for (int i = 0; i < this.nout; i++) {
            outputs[i] = this.neurons[i].forward(inputs);
        }
        return outputs;
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (int i = 0; i < this.nout; i++) {
            params.addAll(this.neurons[i].parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer with ").append(nout).append(" neurons:\n");
        for (int i = 0; i < nout; i++) {
            sb.append("  Neuron ").append(i + 1).append(": Weights = [");
            for (int j = 0; j < nin; j++) {
                sb.append(neurons[i].weights[j].data);
                if (j < nin - 1) {
                    sb.append(", ");
                }
            }
            sb.append("], Bias = ").append(neurons[i].bias.data).append("\n");
        }
        return sb.toString();
    }
}
