package Common.Feature_extraction.Service;

import java.awt.image.BufferedImage;
import java.util.HashMap;

import Common.Concept;
import Common.Feature_extraction.Model.Sign;

public class SignService {
    public static void preprocessing(Sign sign) {
        ImageService imageService = new ImageService();
        BufferedImage img = imageService.ImageLoader(sign.getPath());
        if (img == null) {
            System.out.println("preprocessing: could not load image for path: " + sign.getPath());
            sign.setImage(null);
            return;
        }  
        sign.setImage(img);
        BufferedImage cropped = imageService.ImageCropped(sign.getImage());
        if (cropped == null) {
            System.out.println("preprocessing: cropping returned null, keeping original image for: " + sign.getPath());
        } else {
            sign.setImage(cropped);
        }
    }

    public static void extractFeatures(Sign sign) {
        ImageService imageService = new ImageService();
        if (sign.getImage() == null) {
            System.out.println("extractFeatures: no image available for sign, skipping feature extraction: " + sign.getPath());
            sign.setColorPercentages(new HashMap<>());
            return;
        }
        sign.setColorPercentages(imageService.calculateColorPercentages(sign.getImage()));
    }

    public static Concept extractConcept(Sign sign) {
        String path = sign.getPath().replace("\\", "/");
        String[] parts = path.split("/");
        String conceptString = parts.length > 2 ? parts[parts.length - 4] : path;
        switch (conceptString) {
            case "206 - Stop":
                return Concept.Stop;
            case "102 - Vorfahrt von rechts":
                return Concept.RightOfWayFromRight;
            case "205 - Vorfahrt gewähren":
                return Concept.Yield;
            case "209 - Fahrtrichtung links":
                return Concept.DirectionLeft;
            case "209 - Fahrtrichtung rechts":
                return Concept.DirectionRight;
            case "306 - Vorfahrtsstraße":
                return Concept.PriorityRoad;
            default:
                return Concept.Unknown;
        }
    }
}
