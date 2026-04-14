package Common;

import java.util.List;

public interface Classifier {
    void train(List<VectorData> data);

    Concept classify (float[] fs);

    String getName();
}
