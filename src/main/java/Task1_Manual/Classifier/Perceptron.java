package Task1_Manual.Classifier;

import java.util.List;

import Common.Classifier;
import Common.Concept;
import Common.VectorData;

public class Perceptron implements Classifier{

    private int numFeatures;
    private int numClasses;
    private float[][] weights;
    private float bias = 1;
    private Concept[] classList;

    public Perceptron(int count) {
        this.numFeatures = count;

        int classCount = 0;
        for (Concept c : Concept.values()) {
            if (c != Concept.Unknown) {
                classCount++;
            }
        }
        this.numClasses = classCount;

        this.classList = new Concept[this.numClasses];
        int i = 0;
        for (Concept c : Concept.values()) {
            if (c != Concept.Unknown) {
                this.classList[i] = c;
                i++;
            }
        }

        this.weights = new float[this.numClasses][this.numFeatures + 1];

        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < this.numClasses; i++) {
            for (int j = 0; j < (this.numFeatures+1); j++) {
                this.weights[i][j] = (float) Math.random() * 2 - 1;
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

    private float[] getScores(float[] features) {
        float[] scores = new float[this.numClasses];
        for (int i = 0; i < this.numClasses; i++) {
            scores[i] = this.bias * weights[i][0];
            for (int j = 0; j < this.numFeatures; j++) {
                scores[i] += this.weights[i][j + 1] * features[j];
            }
        }
        return scores;
    }

    private int getMaxConceptIndex(float[] scores) {
        double maxScore = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void train(List<VectorData> trainData) {
        if (trainData == null || trainData.isEmpty()) {
            throw new IllegalArgumentException("Traindata is empty!");
        }

        final int maxEpochs = 1000;
        boolean converged = false;

        for (int epoch = 0; epoch < maxEpochs && !converged; epoch++) {
            int mistakesInEpoch = 0;

            for (VectorData dataPoint : trainData) {
                if (dataPoint.getConcept() == Concept.Unknown) continue;

                float[] features = dataPoint.getValues();
                int predictedIndex = getMaxConceptIndex(getScores(features));
                int correctIndex = getRowIndex(dataPoint.getConcept());

                Concept predictedConcept = classify(features);
                if (predictedConcept != dataPoint.getConcept()) {
                    mistakesInEpoch++;
                }

                for (int j = 0; j < this.numClasses; j++) {
                    
                    if (j == correctIndex && j != predictedIndex) { // false-negative
                        mistakesInEpoch++;
                        this.weights[j][0] += bias; // Bias update
                        for (int k = 0; k < (this.numFeatures); k++) {
                            this.weights[j][k+1] += features[k];
                        }
                    } 
                    else if (j != correctIndex && j == predictedIndex) { // Falsch-Positiv
                        mistakesInEpoch++;
                        this.weights[j][0] -= bias; // Bias update
                        for (int k = 0; k < this.numFeatures; k++) {
                            this.weights[j][k+1] -= features[k];
                        }
                    }
                }
            }
            
            if (mistakesInEpoch == 0) {
                converged = true;
                System.out.println("INFO: Perceptron converged after epoch " + (epoch + 1));
            }
        }
        
        if (!converged) {
             System.out.println("WARNUNG: Perceptron not converged after " + maxEpochs + " epochs!");
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

    @Override
    public String getName() {
        return "Perceptron";
    }
}