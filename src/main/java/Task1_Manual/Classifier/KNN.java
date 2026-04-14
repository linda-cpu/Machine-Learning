package Task1_Manual.Classifier;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import Common.Classifier;
import Common.Concept;
import Common.VectorData;

public class KNN implements Classifier {
    List<VectorData> trainData;
    int k;

    public KNN (int k) {
        this.k = k;
    }

    private float manhattanDistance(float[] a, float[] b) {
    if (a.length != b.length) {
        throw new IllegalArgumentException("Vectors have different lengths!");
    }
    float sum = 0.0f;
    for (int i = 0; i < a.length; i++) {
        sum += (float) Math.abs(a[i] - b[i]);
    }
    return sum;
}


    public Concept classify(float[] features) {
        if(trainData.isEmpty()){
            System.err.println("KNN model has not been trained.");
        }
        List<Float> distances = trainData.stream()
                .map(v -> manhattanDistance(v.getValues(), features))
                .collect(Collectors.toList());

        List<Integer> sortedIndices = IntStream.range(0, distances.size())
                .boxed()
                .sorted(Comparator.comparingDouble(distances::get))
                .toList();

        List<Concept> nearestConcepts = IntStream.range(0, Math.min(k, sortedIndices.size()))
                .mapToObj(i -> trainData.get(sortedIndices.get(i)).getConcept())
                .collect(Collectors.toList());

        return majorityVote(nearestConcepts);
    }

    private Concept majorityVote(List<Concept> neighbors) {
        Map<Concept, Long> freqMap = new HashMap<>();
        for (Concept concept : neighbors) {
            freqMap.put(concept, freqMap.getOrDefault(concept, 0L) + 1);
        }

        return freqMap.entrySet().stream()
                .max(Map.Entry.<Concept, Long>comparingByValue()
                        .thenComparing(Map.Entry.comparingByKey()))
                .map(Map.Entry::getKey)
                .orElse(Concept.Unknown);
    }


    @Override
    public void train(List<VectorData> data) {
        this.trainData = data;
    }

    @Override
    public String getName() {
        return "K-Nearest-Neigbour";
    }
}
