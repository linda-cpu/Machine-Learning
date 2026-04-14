package Task2_DeepLearning;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import Common.Classifier;
import Common.Concept;
import Common.DJLHelper;
import Common.VectorData;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

public class CNN implements Classifier {

    NDManager manager;
    private Predictor<float[], Concept> predictor;
    private Model loadedModel;

    public CNN(NDManager manager) {
        this.manager = manager;
    }

    private SequentialBlock getCnnArchitecture(int outputSize) {
        SequentialBlock cnn = new SequentialBlock();

        cnn.add(new LambdaBlock(x -> new NDList(x.singletonOrThrow().reshape(new Shape(-1, 3, 32, 32)))));

        cnn.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(32)
                .optPadding(new Shape(1, 1))
                .build())
           .add(BatchNorm.builder().build())
           .add(Activation.reluBlock())
           .add(Pool.maxPool2dBlock(new Shape(2, 2)));

        cnn.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(64)
                .optPadding(new Shape(1, 1))
                .build())
           .add(BatchNorm.builder().build())
           .add(Activation.reluBlock())
           .add(Pool.maxPool2dBlock(new Shape(2, 2)));
        
        cnn.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(128)
                .optPadding(new Shape(1, 1))
                .build())
           .add(BatchNorm.builder().build())
           .add(Activation.reluBlock())
           .add(Pool.maxPool2dBlock(new Shape(2, 2)));

        cnn.add(Blocks.batchFlattenBlock())
           .add(Linear.builder().setUnits(128).build())
           .add(Activation.reluBlock())
           .add(Dropout.builder().optRate(0.5f).build())
           .add(Linear.builder().setUnits(outputSize).build());

        return cnn;
    }

    @Override
    public void train(List<VectorData> data) {
        try (NDManager manager = NDManager.newBaseManager()) {
            int inputSize = data.get(0).getValues().length;
            int outputSize = Concept.values().length;

            Dataset dataset = DJLHelper.createDatasetFromVectors(manager, data);
            
            SequentialBlock cnnBlock = getCnnArchitecture(outputSize);

            try (Model model = Model.newInstance("my-cnn-model")) {
                model.setBlock(cnnBlock);

                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                		.addEvaluator(new Accuracy())
                	    .addTrainingListeners(new EvaluatorTrainingListener(), new TrainingListener() {
                	    	private int epochCount = 0;
                	        @Override
                	        public void onEpoch(Trainer trainer) {
                	        	epochCount++;
                	            double acc = trainer.getTrainingResult().getTrainEvaluation("Accuracy")*100;
                	            double loss = trainer.getTrainingResult().getTrainEvaluation("SoftmaxCrossEntropyLoss");
                	            System.out.printf("Epoch %d finished --> Accuracy: %.2f | Loss: %.4f%n", epochCount, acc, loss);
                	        }
                	        @Override public void onTrainingBegin(Trainer trainer) {}
                	        @Override public void onTrainingEnd(Trainer trainer) {}
                	        @Override public void onTrainingBatch(Trainer trainer, BatchData batchData) {}
							@Override public void onValidationBatch(Trainer trainer, BatchData batchData) {}
                	    });

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.initialize(new Shape(1, inputSize));
                    trainer.setMetrics(new Metrics());

                    System.out.println("Starting CNN training with " + data.size() + " images...");

                    EasyTrain.fit(trainer, 35, dataset, null);
                    
                    Path modelDir = Paths.get("build/model/cnn");
                    Files.createDirectories(modelDir);
                    model.save(modelDir, "cnn_optimized"); 
                    System.out.println("\nCNN model saved in: " + modelDir.toAbsolutePath());
                }
                
                if (this.predictor != null) {
                    this.predictor.close();
                    this.predictor = null;
                }
                if (this.loadedModel != null) {
                    this.loadedModel.close();
                    this.loadedModel = null;
                }
            }
        } catch (IOException | TranslateException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Concept classify(float[] features) {
        if (this.predictor == null) {
            try {
                Path modelDir = Paths.get("build/model/cnn");
                this.loadedModel = Model.newInstance("cnn_optimized");
                
                int outputSize = Concept.values().length;

                this.loadedModel.setBlock(getCnnArchitecture(outputSize));

                this.loadedModel.load(modelDir, "cnn_optimized");

                this.predictor = loadedModel.newPredictor(new VectorTranslator());
                
            } catch (Exception e) {
                System.err.println("Error loading the CNN model for prediction: " + e.getMessage());
                e.printStackTrace();
                return Concept.Unknown;
            }
        }

        try {
            float[] floatFeatures = new float[features.length];
            for (int i = 0; i < features.length; i++) {
                floatFeatures[i] = (float) features[i];
            }

            return this.predictor.predict(floatFeatures);
        } catch (TranslateException e) {
            e.printStackTrace();
            return Concept.Unknown;
        }
    }

    @Override
    public String getName() {
        return "Convolutional Neural Network";
    }
}