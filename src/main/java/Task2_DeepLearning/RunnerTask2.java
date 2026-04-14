package Task2_DeepLearning;

import java.util.ArrayList;
import java.util.List;

import Common.Evaluator;
import Common.VectorData;
import Common.Feature_extraction.Service.JsonService;
import Common.Feature_extraction.Service.SignLoader;
import ai.djl.ndarray.NDManager;


public class RunnerTask2 {
    public static void main(String[] args) throws Exception {
    	/*SignLoader signLoader = new SignLoader();
        signLoader.generatFeatureDataset();*/

        JsonService jsonService = new JsonService();
        List<VectorData> allData = new ArrayList<VectorData>(jsonService.setTrainingFromFile("src/main/resources/json/traffic.json"));

        NDManager manager = NDManager.newBaseManager();

        MLP mlp = new MLP(manager);
        CNN cnn = new CNN(manager);
        Evaluator evaluator = new Evaluator();
        evaluator.evaluateModel(cnn, allData, 1, false);
        evaluator.evaluateModel(mlp, allData, 1, false);

    }
}
