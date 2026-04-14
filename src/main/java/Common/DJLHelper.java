package Common;

import java.util.List;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;

public class DJLHelper {
    public static ArrayDataset createDatasetFromVectors(NDManager manager, List<VectorData> dataList) {
        
    	System.out.println("Starting conversion for DJL...");

        int size = dataList.size();
        int inputSize = 32 * 32*3; 

        float[][] rawData = new float[size][inputSize];
        int[] labels = new int[size];

        for (int i = 0; i < size; i++) {
            VectorData vd = dataList.get(i);
            float[] originalValues = vd.getValues(); 
            for (int j = 0; j < inputSize; j++) {
                 if (j < originalValues.length) {
                     rawData[i][j] = (float) originalValues[j]; 
                 }
            }
            labels[i] = vd.getConcept().ordinal();
        }
        NDArray dataND = manager.create(rawData);
        NDArray labelND = manager.create(labels);
        
        return new ArrayDataset.Builder()
                .setData(dataND)
                .optLabels(labelND)
                .setSampling(32, true)
                .build();
    }
}