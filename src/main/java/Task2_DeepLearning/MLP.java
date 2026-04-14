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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.nn.Blocks;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.Activation;

public class MLP implements Classifier {
    NDManager manager;
    private Predictor<float[], Concept> predictor;
    private Model loadedModel;

    public MLP(NDManager manager) {
        this.manager = manager;
    }

    private SequentialBlock getMlpArchitecture(long inputSize, int outputSize) {
        SequentialBlock mlpBlock = new SequentialBlock();

        mlpBlock.add(Blocks.batchFlattenBlock(inputSize));

        mlpBlock.add(Linear.builder().setUnits(1024).build());
        mlpBlock.add(BatchNorm.builder().build());
        mlpBlock.add(Activation.reluBlock());
        mlpBlock.add(Dropout.builder().optRate(0.5f).build());

        mlpBlock.add(Linear.builder().setUnits(512).build());
        mlpBlock.add(BatchNorm.builder().build());
        mlpBlock.add(Activation.reluBlock());
        mlpBlock.add(Dropout.builder().optRate(0.3f).build());

        mlpBlock.add(Linear.builder().setUnits(256).build());
        mlpBlock.add(Activation.reluBlock());
        
        mlpBlock.add(Linear.builder().setUnits(128).build());
        mlpBlock.add(Activation.reluBlock());
        
        mlpBlock.add(Linear.builder().setUnits(outputSize).build());
        
        return mlpBlock;
    }

    @Override
    public void train(List<VectorData> data) {
        try (NDManager manager = NDManager.newBaseManager()) {
        	if (data == null || data.isEmpty()) {
                System.err.println("Training data is empty!");
                return;
            }
        	int inputSize = data.get(0).getValues().length;
            int outputSize = Concept.values().length;

            Dataset dataset = DJLHelper.createDatasetFromVectors(manager, data);
            SequentialBlock mlpBlock = getMlpArchitecture(inputSize, outputSize);

            try (Model model = Model.newInstance("my-mlp-model")) {
                model.setBlock(mlpBlock);

                DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                		.optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build())
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

                    System.out.println("Starting MLP training with " + data.size() + " vectors...");
                    
                    EasyTrain.fit(trainer, 50, dataset, null);
                    
                    Path modelDir = Paths.get("build/model/mlp");
                    Files.createDirectories(modelDir);
                    model.save(modelDir, "mlp_optimized"); 
                    System.out.println("\nModel saved in: " + modelDir.toAbsolutePath());
                    
                    System.out.println("Training finished! Accuracy: " + 
                        trainer.getTrainingResult().getTrainEvaluation("Accuracy"));
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
                Path modelDir = Paths.get("build/model/mlp");
                this.loadedModel = Model.newInstance("mlp_optimized");

                long inputSize = features.length;
                int outputSize = Concept.values().length;
                
                this.loadedModel.setBlock(getMlpArchitecture(inputSize, outputSize));
                
                this.loadedModel.load(modelDir, "mlp_optimized");
                
                this.predictor = loadedModel.newPredictor(new VectorTranslator());
                
            } catch (Exception e) {
            	System.err.println("Error loading the MLP model for prediction: " + e.getMessage());
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
        return "MultilayerPerceptron";
    }
}