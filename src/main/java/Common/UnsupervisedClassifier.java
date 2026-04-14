package Common;

import java.util.List;

public interface UnsupervisedClassifier extends Classifier {
    float getCalinskiHarabasz(List<VectorData> data);
    float getSillhouetteScore(List<VectorData> data);

}
