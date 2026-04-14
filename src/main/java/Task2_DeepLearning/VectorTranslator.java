package Task2_DeepLearning;

import Common.Concept;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class VectorTranslator implements Translator<float[], Concept> {

    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        NDArray array = ctx.getNDManager().create(input);        
        return new NDList(array);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK; 
    }

    @Override
    public Concept processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray output = list.get(0);
        long bestIndex = output.softmax(0).argMax().getLong();
        if (bestIndex >= 0 && bestIndex < Concept.values().length) {
            return Concept.values()[(int) bestIndex];
        }
        return Concept.Unknown;
    }
}