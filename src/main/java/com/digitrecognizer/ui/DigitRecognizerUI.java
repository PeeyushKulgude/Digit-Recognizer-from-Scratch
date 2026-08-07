package com.digitrecognizer.ui;

import com.digitrecognizer.models.MLP;
import com.digitrecognizer.utils.TrainingConfig;
import com.digitrecognizer.utils.Value;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import javax.swing.*;

/**
 * Very small, self-contained UI:
 *   - draw a digit with the mouse on a 280x280 canvas
 *   - click "Predict" to downsample to 28x28, run it through the latest
 *     saved MLP checkpoint, and show the predicted label + confidence
 *   - click "Clear" to reset the canvas
 *
 * Run it directly:
 *   javac -d out $(find src -name "*.java")
 *   java -cp out com.digitrecognizer.ui.DigitRecognizerUI
 *
 * NOTE on normalization: this assumes MnistDataLoader feeds the network
 * pixel values scaled to [0, 1] (pixel / 255.0), which is the standard
 * MNIST convention. If your loader normalizes differently (e.g. mean/std
 * standardization), adjust toNetworkInput() below to match, or predictions
 * will be systematically wrong even though the code runs fine.
 */
public class DigitRecognizerUI extends JFrame {

    private static final int CANVAS_SIZE = 280; // 10x scale of 28x28
    private static final int GRID_SIZE = 28;
    private static final int STROKE_WIDTH = 18;

    private final BufferedImage canvasImage;
    private final Graphics2D canvasGraphics;
    private final JLabel resultLabel = new JLabel("Draw a digit, then click Predict.");
    private MLP mlp;

    public DigitRecognizerUI() throws IOException {
        super("Digit Recognizer");

        canvasImage = new BufferedImage(CANVAS_SIZE, CANVAS_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        canvasGraphics = canvasImage.createGraphics();
        canvasGraphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        clearCanvas();

        JPanel drawPanel = buildDrawPanel();

        JButton predictButton = new JButton("Predict");
        predictButton.addActionListener(e -> predict());

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> {
            clearCanvas();
            drawPanel.repaint();
            resultLabel.setText("Draw a digit, then click Predict.");
        });

        resultLabel.setFont(resultLabel.getFont().deriveFont(Font.BOLD, 18f));
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);

        JPanel buttonPanel = new JPanel(new FlowLayout());
        buttonPanel.add(predictButton);
        buttonPanel.add(clearButton);

        setLayout(new BorderLayout(8, 8));
        add(drawPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
        add(resultLabel, BorderLayout.NORTH);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setResizable(false);
        setLocationRelativeTo(null);

        loadModelOrWarn();
    }

    private JPanel buildDrawPanel() {
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(canvasImage, 0, 0, null);
            }
        };
        panel.setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
        panel.setBorder(BorderFactory.createLineBorder(Color.GRAY));

        MouseAdapter drawingHandler = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                paintDot(e.getX(), e.getY());
                panel.repaint();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                paintDot(e.getX(), e.getY());
                panel.repaint();
            }
        };
        panel.addMouseListener(drawingHandler);
        panel.addMouseMotionListener(drawingHandler);

        return panel;
    }

    private void paintDot(int x, int y) {
        canvasGraphics.setColor(Color.WHITE);
        canvasGraphics.fill(new Ellipse2D.Double(
                x - STROKE_WIDTH / 2.0, y - STROKE_WIDTH / 2.0, STROKE_WIDTH, STROKE_WIDTH));
    }

    private void clearCanvas() {
        canvasGraphics.setColor(Color.BLACK);
        canvasGraphics.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }

    /** Loads the most recently saved .model checkpoint, same convention Runner.java uses. */
    private void loadModelOrWarn() throws IOException {
        File dir = new File(TrainingConfig.getSavedModelsDir());
        File[] modelFiles = dir.exists() ? dir.listFiles((d, name) -> name.endsWith(".model")) : null;

        if (modelFiles == null || modelFiles.length == 0) {
            resultLabel.setText("No trained model found in " + TrainingConfig.getSavedModelsDir());
            return;
        }

        File latest = java.util.Arrays.stream(modelFiles)
                .max(Comparator.comparingLong(File::lastModified))
                .orElse(null);

        mlp = MLP.load(latest.getPath());
        resultLabel.setText("Loaded model: " + latest.getName() + ". Draw a digit, then click Predict.");
    }

    /** Downsamples the 280x280 canvas to a 784-length, normalized 28x28 input vector. */
    private double[] toNetworkInput() {
        Image scaled = canvasImage.getScaledInstance(GRID_SIZE, GRID_SIZE, Image.SCALE_SMOOTH);
        BufferedImage small = new BufferedImage(GRID_SIZE, GRID_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = small.createGraphics();
        g.drawImage(scaled, 0, 0, null);
        g.dispose();

        double[] input = new double[GRID_SIZE * GRID_SIZE];
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                int gray = small.getRaster().getSample(col, row, 0); // 0-255
                input[row * GRID_SIZE + col] = gray / 255.0;
            }
        }
        return input;
    }

    private void predict() {
        if (mlp == null) {
            resultLabel.setText("No model loaded — train one first.");
            return;
        }

        double[] input = toNetworkInput();
        Value[] outputs = mlp.forward(input);
        Value[] probabilities = mlp.softmax(outputs);

        int predictedLabel = -1;
        double confidence = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i].data > confidence) {
                confidence = probabilities[i].data;
                predictedLabel = i;
            }
        }

        resultLabel.setText(String.format("Prediction: %d (confidence: %.1f%%)",
                predictedLabel, confidence * 100));
    }
}