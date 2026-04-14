package Common.Feature_extraction.Service;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.google.gson.stream.JsonWriter;

import Common.Feature_extraction.Model.Sign;
import Common.Feature_extraction.Model.SignList;

public class SignLoader {
    private final String[] IMAGE_EXTENSIONS = { ".bmp", ".png", ".jpg", ".jpeg", ".webp", ".gif" };

    private SignList loadAllSigns(String imgFolderPath) throws IOException {
        SignList signs = new SignList();
        Path imgPath = Paths.get(imgFolderPath);

        Files.walk(imgPath)
                .filter(Files::isRegularFile)
                .filter(p -> isImageFile(p))
                .forEach(p -> {
                    String conceptName = imgPath.relativize(p).toString();
                    Sign sign = new Sign(imgPath.toString() + "/" + conceptName, null);
                    signs.addSign(sign);
                });

        return signs;
    }

    public void generatSignDataset() throws IOException {
        SignList allSigns = loadAllSigns("src/main/resources/img/");
        SignList processedSigns = new SignList();

        for (Sign sign : allSigns.getSigns()) {
            SignService.preprocessing(sign);
            SignService.extractFeatures(sign);
            processedSigns.addSign(sign);
        }
        JsonService jsonService = new JsonService();
        jsonService.prepareSignForJSON(processedSigns);
    }

    public void generatFeatureDataset() throws IOException {
        ImageService imageService = new ImageService();
        SignList allSigns = loadAllSigns("src/main/resources/img/");

        try (JsonWriter writer = new JsonWriter(new FileWriter("src/main/resources/json/traffic.json"))) {
        	writer.beginObject();
        	writer.name("author").value("Linda");
            
            writer.name("properties");
            writer.beginArray();
            writer.value("pixels");
            writer.endArray();

            writer.name("vectors");
            writer.beginArray();

            for (Sign sign : allSigns.getSigns()) {
                try {
                    SignService.preprocessing(sign);

                    double[] vectorValues = imageService.imageToVector(sign.getImage(), 32, 32);

                    writer.beginObject();
                    writer.name("concept").value(sign.getConcept().toString());
                    writer.name("values");
                    writer.beginArray();
                    if (vectorValues != null) {
                        for (double v : vectorValues) {
                            writer.value(v);
                        }
                    }
                    writer.endArray();
                    writer.endObject();

                    sign.setImage(null); 

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            writer.endArray();
            writer.endObject();
        }
    }

    private boolean isImageFile(Path path) {
        String lower = path.toString().toLowerCase();
        for (String ext : IMAGE_EXTENSIONS) {
            if (lower.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }
}
