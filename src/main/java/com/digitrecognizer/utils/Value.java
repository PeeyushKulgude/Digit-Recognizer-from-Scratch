package com.digitrecognizer.utils;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class Value {

    public double data;
    double grad;
    Set<Value> prev;
    Function<Void, Void> _backward;
    List<Value> topologicalList = null;

    public Value(double data) {
        this.data = data;
        this.grad = 0.0;
        this.prev = new HashSet<>();
        this._backward = null;
    }

    public Value(double data, Set<Value> prev) {
        this.data = data;
        this.grad = 0.0;
        this.prev = prev;
    }

    public void setGradZero() {
        this.grad = 0.0;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, Set.of(this, other));
        out._backward = (Void v) -> {
            this.grad += out.grad;
            other.grad += out.grad;
            return null;
        };
        return out;
    }

    public Value multiply(Value other) {
        Value out = new Value(this.data * other.data, Set.of(this, other));
        out._backward = (Void v) -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
            return null;
        };
        return out;
    }

    public Value divide(Value other) {
        Value out = new Value(this.data / other.data, Set.of(this, other));
        out._backward = (Void v) -> {
            this.grad += (1.0 / other.data) * out.grad;
            other.grad += (-this.data / (other.data * other.data)) * out.grad;
            return null;
        };
        return out;
    }

    public Value tanh() {
        double t = Math.tanh(this.data);
        Value out = new Value(t, Set.of(this));
        out._backward = (Void v) -> {
            this.grad += (1 - t * t) * out.grad;
            return null;
        };
        return out;
    }

    public Value exp() {
        double e = Math.exp(this.data);
        Value out = new Value(e, Set.of(this));
        out._backward = (Void v) -> {
            this.grad += e * out.grad;
            return null;
        };
        return out;
    }

    public Value log() {
        if (this.data <= 0) {
            throw new IllegalArgumentException("Logarithm only defined for positive numbers");
        }
        Value out = new Value(Math.log(this.data), Set.of(this));
        out._backward = (Void v) -> {
            this.grad += (1 / this.data) * out.grad;
            return null;
        };
        return out;
    }

    public void backward() {
        if (topologicalList == null) {
            buildTopologicalList();
        }

        this.grad = 1.0; // Seed the gradient for the output node
        for (Value v : this.topologicalList) {
            if (v._backward != null) {
                v._backward.apply(null);
            }
        }
    }

    private void buildTopologicalList() {
        List<Value> topoList = new java.util.ArrayList<>();
        Set<Value> visited = new HashSet<>();

        buildTopologicalListHelper(this, visited, topoList);
        this.topologicalList = topoList.reversed();
    }

    private void buildTopologicalListHelper(Value v, Set<Value> visited, List<Value> topoList) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.prev) {
                buildTopologicalListHelper(child, visited, topoList);
            }
            topoList.add(v);
        }
    }

    public void step(double learningRate) {
        this.data -= learningRate * this.grad;
    }

    @Override
    public String toString() {
        return "Value{"
                + "data=" + data
                + ", grad=" + grad
                + '}';
    }
}
