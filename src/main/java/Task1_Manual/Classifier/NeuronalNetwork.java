package Task1_Manual.Classifier;

import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.List;

import Common.Classifier;
import Common.Concept;
import Common.VectorData;

public class NeuronalNetwork implements Classifier {

    private int numNodesIn;
    private int numNodesOut;
    private float[][] weights;
    private float bias = 1;
    private Concept[] classList;
    private float learningRate = 0.1f;
    private int epochs = 200;

    public NeuronalNetwork(int numNodesIn) {
        this.numNodesIn = numNodesIn;

        int classCount = 0;
        for (Concept c : Concept.values()) {
            if (c != Concept.Unknown) {
                classCount++;
            }
        }
        this.numNodesOut = classCount;

        this.classList = new Concept[this.numNodesOut];
        int i = 0;
        for (Concept c : Concept.values()) {
            if (c != Concept.Unknown) {
                this.classList[i] = c;
                i++;
            }
        }

        this.weights = new float[this.numNodesOut][this.numNodesIn + 1];

        initializeWeights();
    }

    void initializeWeights() {
        for (int i = 0; i < this.numNodesOut; i++) {
            for (int j = 0; j < (this.numNodesIn+1); j++) {
                this.weights[i][j] = (float) Math.random() * 2f - 1f;
            }
        }
    }

    private int getRowIndex(Concept c) {
        for (int i = 0; i < this.classList.length; i++) {
            if (this.classList[i] == c) {
                return i;
            }
        }
        throw new IllegalArgumentException("Concept not known: " + c);
    }

    private Concept getConceptFromRowIndex(int index) {
        if (index < 0 || index >= this.classList.length) {
            throw new IllegalArgumentException("Index not known: " + index);
        }
        return this.classList[index];
    }

    private float sigmoid(float z) {
        return (float) (1.0f / (1.0f + Math.exp(-z)));
    }

    private float sigmoidDerivative(float output) {
        return output * (1.0f - output);
    }

    private float[] feedForward(float[] features) {
        float[] outputs = new float[this.numNodesOut];
        for (int i = 0; i < this.numNodesOut; i++) {
            outputs[i] = sigmoid(getScores(features)[i]);
        }
        return outputs;
    }

    private float[] getScores(float[] features) {
        float[] scores = new float[this.numNodesOut];
        for (int i = 0; i < this.numNodesOut; i++) {
            scores[i] = this.bias * weights[i][0];
            for (int j = 0; j < this.numNodesIn; j++) {
                scores[i] += this.weights[i][j + 1] * features[j];
            }
        }
        return scores;
    }

    public void train(List<VectorData> trainData) {
        Instant startTime = Instant.now();
        float accuracy = 0.0f;
        if (trainData == null || trainData.isEmpty()) {
            throw new IllegalArgumentException("Trainingsdaten sind leer!");
        }

        boolean converged = false;
       // int epoch = 0;

        for (int epoch = 0; epoch < this.epochs && !converged; epoch++) {
     //while (accuracy < 70 && epoch < 500) {   //Learningrate adjustment
            //Collections.shuffle(trainData);
            //epoch++;
            int mistakesInEpoch = 0;
            for (VectorData v : trainData) {
                if (v.getConcept() == Concept.Unknown)
                    continue;

                float[] features = v.getValues();
                float[] outputs = feedForward(features);
                int correctIndex = getRowIndex(v.getConcept());

                Concept predictedConcept = classify(features);
                if (predictedConcept != v.getConcept()) {
                    mistakesInEpoch++;
                }

                for (int j = 0; j < this.numNodesOut; j++) {
                    float correctOutput = (j == correctIndex) ? 1.0f : 0.0f;
                    float error = correctOutput - outputs[j];
                    float delta = error * sigmoidDerivative(outputs[j]);

                    // Update bias
                    this.weights[j][0] += learningRate * delta * bias;
                    // Update weights
                    for (int k = 0; k < this.numNodesIn; k++) {
                        this.weights[j][k + 1] += learningRate * delta * features[k];
                    }
                }
            }

            if (mistakesInEpoch == 0) {
                converged = true;
                System.out.println("INFO: Perzeptron konvergiert nach Epoche " + (epoch + 1));
            }
            accuracy = (float) (trainData.size() - mistakesInEpoch) / trainData.size() * 100.0f;
        }

        Instant endTime = Instant.now();
        Duration duration = Duration.between(startTime, endTime);
        System.out.println("Lernrate: " + learningRate + "; Laufzeit: " + duration.toMillis() + "; Genauigkeit: " + accuracy + "%.");

        if (!converged) {
            System.out.println("WARNUNG: Perzeptron nach " + epochs + " Epochen nicht konvergiert.");
        }
    }

    public Concept classify(float[] features) {
        float[] scores = getScores(features);

        double maxScore = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;

        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }

        if (maxIndex == -1) {
            return Concept.Unknown;
        }

        return getConceptFromRowIndex(maxIndex);
    }

    public void changeConfig(float bias, float learningRate, int epochs) {
        this.bias = bias;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    @Override
    public String getName() {
        return "NeuronalNetwork";
    }
}