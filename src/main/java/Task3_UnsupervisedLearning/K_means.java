package Task3_UnsupervisedLearning;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import Common.Concept;
import Common.UnsupervisedClassifier;
import Common.VectorData;
import smile.clustering.CentroidClustering;
import smile.clustering.KMeans;

public class K_means implements UnsupervisedClassifier {

    private CentroidClustering<double[], double[]> kMeansModel;
    private Concept[] ClusterConcepts;
    private int numClusters = 5;
    private int maxIter = 100;
    private String jsonName = "";

    K_means(int numClusters, int maxIter, String jsonName) {
        this.numClusters = numClusters;
        this.maxIter = maxIter;
        this.jsonName = jsonName;
        this.ClusterConcepts = new Concept[numClusters];
    }

    @Override
    public void train(List<VectorData> data) {
        int n = data.size();
        Concept[] concepts = new Concept[n];
        int dim = data.get(0).getValues().length;
        double[][] features = new double[n][dim];
        for (int i = 0; i < n; i++) {
            concepts[i] = data.get(i).getConcept();
            for (int j = 0; j < dim; j++) {
                features[i][j] = data.get(i).getValues()[j];
            }
        }
        this.kMeansModel = KMeans.fit(features, numClusters, maxIter);

        System.out.println(this.kMeansModel);

        // Success output
        analyzeClusters(features, concepts);
        System.out.println("Training completed successfully.");
        System.out.println("Model created with " + kMeansModel.size() + " clusters.");
        System.out.printf("Distortion (Error): %.4f%n", kMeansModel.distortion());
    }

    public void analyzeClusters(double[][] features, Concept[] concepts) {
        // Struktur: Map<ClusterID, Map<LabelName, Anzahl>>
        Map<Integer, Map<Concept, Integer>> clusterAnalysis = new HashMap<>();

        for (int k = 0; k < numClusters; k++) {
            clusterAnalysis.put(k, new HashMap<>());
        }

        System.out.println("Analysiere Cluster-Zuweisungen...");

        for (int i = 0; i < features.length; i++) {
            int clusterId = kMeansModel.predict(features[i]);
            Concept concept = concepts[i];
            Map<Concept, Integer> labelCounts = clusterAnalysis.get(clusterId);
            labelCounts.put(concept, labelCounts.getOrDefault(concept, 0) + 1);
        }

        System.out.println("\n--- Cluster Analyse ---");

        for (int k = 0; k < numClusters; k++) {
            Map<Concept, Integer> counts = clusterAnalysis.get(k);

            if (counts.isEmpty()) {
                System.out.println("Cluster " + k + " ist leer.");
                continue;
            }

            // Gewinner ermitteln
            Concept winnerConcept = Concept.Unknown;
            int maxCount = -1;

            System.out.println("Cluster " + k + " enthält:");
            for (Entry<Concept, Integer> entry : counts.entrySet()) {
                System.out.println("   - " + entry.getKey() + ": " + entry.getValue() + " mal");

                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    winnerConcept = entry.getKey();
                    this.ClusterConcepts[k] = winnerConcept;
                }
            }
            System.out.println("=> Interpretation: Cluster " + k + " ist wahrscheinlich '" + winnerConcept + "'");
            System.out.println("-------------------------");
        }
        System.out.println(Arrays.toString(this.ClusterConcepts));
    }

    @Override
    public Concept classify(float[] data) {
        double[] features = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            features[i] = (double) data[i];
        }
        int clusterId = kMeansModel.predict(features);
        if (clusterId >= 0 && clusterId < ClusterConcepts.length) {
            return ClusterConcepts[clusterId];
        }
        return Concept.Unknown;
    }

    @Override
    public String getName() {
        return "K-Means-" + numClusters + "-" + jsonName;
    }

    @Override
    public float getCalinskiHarabasz(List<VectorData> data) {
        int n = data.size();
        int k = numClusters;
        if (n <= k)
            return 0.0f;

        double sse = kMeansModel.distortion();
        double tss = calculateTSS(data);

        double ssb = tss - sse;
        double chIndex = (ssb / (k - 1)) / (sse / (n - k));
        return (float) chIndex;
    }

    @Override
    public float getSillhouetteScore(List<VectorData> data) {
        int sampleSize = Math.min(1000, data.size());
        List<double[]> sampleFeatures = new ArrayList<>();
        int[] sampleLabels = new int[sampleSize];

        for(int i=0; i<sampleSize; i++) {
            double[] feats = convertToDouble(data.get(i).getValues());
            sampleFeatures.add(feats);
            sampleLabels[i] = kMeansModel.predict(feats);
        }

        double totalSilhouette = 0.0;

        for (int i = 0; i < sampleSize; i++) {
            double[] currentPoint = sampleFeatures.get(i);
            int currentCluster = sampleLabels[i];

            double a = 0.0;
            int countSame = 0;

            double[] distToClusters = new double[numClusters];
            int[] countToClusters = new int[numClusters];

            for (int j = 0; j < sampleSize; j++) {
                if (i == j) continue;
                double dist = distance(currentPoint, sampleFeatures.get(j));
                int otherCluster = sampleLabels[j];

                if (otherCluster == currentCluster) {
                    a += dist;
                    countSame++;
                } else {
                    distToClusters[otherCluster] += dist;
                    countToClusters[otherCluster]++;
                }
            }

            if (countSame > 0) a /= countSame;

            double b = Double.MAX_VALUE;
            boolean foundNeighbor = false;
            
            for (int k = 0; k < numClusters; k++) {
                if (k == currentCluster || countToClusters[k] == 0) continue;
                
                double avgDist = distToClusters[k] / countToClusters[k];
                if (avgDist < b) {
                    b = avgDist;
                    foundNeighbor = true;
                }
            }
            if (!foundNeighbor) b = 0.0;

            double s = 0.0;
            if (countSame > 0) {
                s = (b - a) / Math.max(a, b);
            }
            totalSilhouette += s;
        }

        return (float) (totalSilhouette / sampleSize);
    }

    private double[] convertToDouble(float[] fs) {
        double[] ds = new double[fs.length];
        for (int i = 0; i < fs.length; i++) {
            ds[i] = (double) fs[i];
        }
        return ds;
    }

    private double calculateTSS(List<VectorData> data) {
        int dim = data.get(0).getValues().length;
        double[] globalMean = new double[dim];

        for (VectorData vec : data) {
            for (int i = 0; i < dim; i++) {
                globalMean[i] += vec.getValues()[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            globalMean[i] /= data.size();
        }
        double tss = 0.0;
        for (VectorData vec : data) {
            double distSq = 0.0;
            for (int i = 0; i < dim; i++) {
                double diff = vec.getValues()[i] - globalMean[i];
                distSq += diff * diff;
            }
            tss += distSq;
        }
        return tss;
    }

    private double distance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

}
