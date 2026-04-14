package Task1_Manual;
import java.util.ArrayList;
import java.util.List;

import Task1_Manual.Classifier.KNN;
import Task1_Manual.Classifier.NeuronalNetwork;
import Task1_Manual.Classifier.Perceptron;
import Common.VectorData;
import Common.Feature_extraction.Model.Json;
import Common.Evaluator;
import Common.Feature_extraction.Service.JsonService;
import Common.Feature_extraction.Service.SignLoader;

public class RunnerTask1 {
    public static void main(String[] args) throws Exception {
        /*SignLoader signLoader = new SignLoader();
        signLoader.generatSignDataset();*/
        JsonService jsonService = new JsonService();
        List<VectorData> allData = new ArrayList<>(jsonService.setTrainingFromFile("src/main/resources/json/traffic.json"));
        int lengthVector = allData.getFirst().getValues().length;
        int runs = 1;
        NeuronalNetwork neuronalNetwork = new NeuronalNetwork(lengthVector);
        Perceptron perceptron = new Perceptron(lengthVector);
        KNN knn = new KNN(3);
        Evaluator evaluator = new Evaluator();
        evaluator.evaluateModel(neuronalNetwork, allData, runs, false);
        evaluator.evaluateModel(perceptron, allData, runs, false);
        evaluator.evaluateModel(knn, allData, runs, false);

    }
}
