package Common.Feature_extraction.Model;

import java.awt.image.BufferedImage;
import java.util.Collection;
import java.util.Map;
import Common.Concept;
import Common.Feature_extraction.Service.SignService;

public class Sign {
    private String path;
    private String filename;
    private String name;
    private Concept concept;
    private BufferedImage image;
    private Map<String, Float> colorPercentages;

    // learn constructor
    public Sign(String path, SignList signList) {
        setPath(path);
        setConceptFromPath();
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
        filename = path.contains("\\")
                ? path.substring(path.lastIndexOf('\\') + 1)
                : path;
        setFilename(filename);
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
        name = filename.contains(".")
                ? filename.substring(0, filename.lastIndexOf('.'))
                : filename;
        setName(name);
    }

    public Map<String, Float> getColorPercentages() {
        return colorPercentages;
    }

    public float[] getColorPercentageValues() {
    	Collection<Float> values = this.getColorPercentages().values();
    	float[] result = new float[values.size()];
    	int i= 0;
    	for (float v : values) {
    		result[i++] = (float) (Math.round(v * 100.0)/100.0);
    	}
    	return result;
    }

    public void setColorPercentages(Map<String, Float> colorPercentages) {
        this.colorPercentages = colorPercentages;
    }

    public BufferedImage getImage() {
        return image;
    }

    public void setImage(BufferedImage image) {
        this.image = image;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Concept getConcept() {
        return concept;
    }

    public void setConcept(Concept concept) {
        this.concept = concept;
    }

    public void setConceptFromPath() {
        this.concept = SignService.extractConcept(this);
    }
}
