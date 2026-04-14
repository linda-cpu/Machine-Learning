package Task3_UnsupervisedLearning;

import java.util.ArrayList;
import java.util.List;

import Common.Evaluator;
import Common.VectorData;
import Common.Feature_extraction.Service.JsonService;
public class RunnerTask3 {
    public static void main(String[] args) throws Exception {
        /*SignLoader signLoader = new SignLoader();
        signLoader.generatFeatureDataset();*/

        JsonService jsonService = new JsonService();
        List<String> jsonDatas = new ArrayList<String>();
        jsonDatas.add("traffic_Linda.json");
        jsonDatas.add("traffic_Alex.json");
        jsonDatas.add("traffic_Leonie.json");
        jsonDatas.add("traffic_Erik.json");
        jsonDatas.add("traffic_Task2.json");

        for (String jsonName : jsonDatas){
            List<VectorData> allData = new ArrayList<VectorData>(jsonService.setTrainingFromFile("src/main/resources/json/" + jsonName));

            for (int i = 50; i <= 100; i += 2){
                K_means k_means = new K_means(i, 100, jsonName);
                Evaluator evaluator = new Evaluator();
                evaluator.evaluateModel(k_means, allData, 5, true);
            }
        }
    }
}
