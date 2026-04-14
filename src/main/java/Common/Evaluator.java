package Common;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class Evaluator {

    List<Double> accuracies = new ArrayList<>();
    List<Double> chIndices = new ArrayList<>();
    List<Double> silhouetteScores = new ArrayList<>();
    int runs = 0;

    public void evaluateModel(Classifier classifier, List<VectorData> allData, int runs, boolean isUnsupervised) {
        this.runs = runs;
        accuracies.clear();
        chIndices.clear();
        silhouetteScores.clear();
        System.out.println(classifier.getName() + ": ");

        for (int run = 0; run < runs; run++) {
            Collections.shuffle(allData, new Random(System.currentTimeMillis() + run));

            int trainSize = (int) (allData.size() * 0.7);
            List<VectorData> trainData = new ArrayList<>(allData.subList(0, trainSize));
            List<VectorData> testData = new ArrayList<>(allData.subList(trainSize, allData.size()));

            classifier.train(trainData);

            // Unsupervised evaluation
            if (isUnsupervised) {
                UnsupervisedClassifier unsupervisedClassifier = (UnsupervisedClassifier) classifier;

                double chIndex = unsupervisedClassifier.getCalinskiHarabasz(trainData);
                double silhouetteScore = unsupervisedClassifier.getSillhouetteScore(trainData);

                chIndices.add(chIndex);
                silhouetteScores.add(silhouetteScore);

                System.out.printf("   Calinski-Harabasz: %.2f | Silhouette: %.4f\n", chIndex, silhouetteScore);
            }

            // Supervised evaluation
            float acc = testSingleRun(classifier, testData);
            accuracies.add((double) acc);
            System.out.printf(" Run %d: Accuracy = %.2f%%%n", run + 1, acc * 100);
            printStats(classifier.getName(), isUnsupervised);

        }
    }

    private float testSingleRun(Classifier model, List<VectorData> testData) {
        int correct = 0;
        for (VectorData v : testData) {
            Concept prediction = model.classify(v.getValues());
            if (prediction == v.getConcept()) {
                correct++;
            }
        }
        return (float) correct / testData.size();
    }

    private void printStats(String modelName, boolean isUnsupervised) {
        double meanAcc = calculateMean(accuracies);
        double[] AccCI = calulateConfidenceInterval(accuracies, meanAcc);
        /* double stdAcc = calculateStdDev(accuracies, meanAcc);
        double margin = (accuracies.size() > 1) ? 1.96 * stdAcc / Math.sqrt(accuracies.size()) : 0.0;
        double ciLow = meanAcc - margin;
        double ciHigh = meanAcc + margin; */

        // 2. Unsupervised Stats berechnen (nur wenn vorhanden)
        double meanCH = isUnsupervised ? calculateMean(chIndices) : 0.0;
        double[] ChCI = isUnsupervised ? calulateConfidenceInterval(chIndices, meanCH) : new double[]{0.0, 0.0};
        double meanSil = isUnsupervised ? calculateMean(silhouetteScores) : 0.0;
        double[] SilCI = isUnsupervised ? calulateConfidenceInterval(silhouetteScores, meanSil) : new double[]{0.0, 0.0};

        // Konsole Output
        /*System.out.println("\n--- Statistics ---");
        System.out.printf("Accuracy: %.2f%%\n", meanAcc * 100);
        if (accuracies.size() > 1) {
            System.out.printf("95%% CI: [%.2f%% – %.2f%%]\n", ciLow * 100, ciHigh * 100);
        }
        if (isUnsupervised) {
            System.out.printf("Mean CH-Index: %.2f\n", meanCH);
            System.out.printf("Mean Silhouette: %.4f\n", meanSil);
        }*/

        // CSV Speichern mit neuen Parametern
        saveToCsv(modelName, meanAcc, AccCI[0], AccCI[1], meanCH, ChCI[0], ChCI[1], meanSil, SilCI[0], SilCI[1], runs);
    }

    private void saveToCsv(String modelName, double meanAcc, double ciLow, double ciHigh,
                           double meanCH, double ciLowCH, double ciHighCH,
                           double meanSil, double ciLowSil, double ciHighSil,
                           int runs) {
        String filepath = "ergebnisse.csv";
        boolean fileExists = new java.io.File(filepath).exists();

        try (PrintWriter writer = new PrintWriter(new FileWriter(filepath, true))) {
            if (!fileExists) {
                writer.println("Timestamp;Model;Runs;Accuracy;CI_Low;CI_High;CH-Index;CI_Low_CH;CI_High_CH;Silhouette;CI_Low_Sil;CI_High_Sil");
            }

            String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
            writer.printf("%s;%s;%d;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f;%.4f;%.4f;%.4f%n",
                    timeStamp, modelName, runs,
                    meanAcc * 100, ciLow * 100, ciHigh * 100,
                    meanCH, ciLowCH, ciHighCH,
                    meanSil, ciLowSil, ciHighSil);

            System.out.println("-> Statistik saved in '" + filepath + "'.");

        } catch (IOException e) {
            System.err.println("Error while saving: " + e.getMessage());
        }
    }   

    private double[] calulateConfidenceInterval(List<Double> values, double mean) {
        double stdDev = calculateStdDev(values, mean);
        double margin = (values.size() > 1) ? 1.96 * stdDev / Math.sqrt(values.size()) : 0.0;
        return new double[]{mean - margin, mean + margin};
    }

    private double calculateMean(List<Double> values) {
        return values.stream().mapToDouble(x -> x).average().orElse(0.0);
    }

    private double calculateStdDev(List<Double> values, double mean) {
        if (values.size() <= 1)
            return 0.0;
        return Math.sqrt(values.stream().mapToDouble(x -> Math.pow(x - mean, 2)).average().orElse(0.0));
    }
}
