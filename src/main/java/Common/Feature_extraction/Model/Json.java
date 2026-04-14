package Common.Feature_extraction.Model;

import java.util.List;

import Common.VectorData;

public class Json {
    private String author;
    private List<String> properties;
    private List<VectorData> vectors;

    public String getAuthor() { return author; }
    public void setAuthor(String author) { this.author = author; }

    public List<String> getProperties() { return properties; }
    public void setProperties(List<String> properties) { this.properties = properties; }

    public List<VectorData> getVectors() { return vectors; }
    public void setVectors(List<VectorData> vectors) { this.vectors = vectors; }
}