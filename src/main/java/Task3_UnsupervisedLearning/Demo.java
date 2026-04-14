package Task3_UnsupervisedLearning;

import smile.clustering.KMeans;
import smile.clustering.CentroidClustering;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

public class Demo {

    // PFAD ANPASSEN: Wähle ein interessantes Bild (z.B. Stoppschild im Schatten)
    static String IMAGE_PATH = "src/main/resources/img/306 - Vorfahrtsstraße/0/80x60+2/X0Y0.jpg";
    static String IMAGE_PATH2 = "src/main/resources/img/DemoTask3/Garten.jpg";

    public static void main(String[] args) throws Exception {
        // 1. Bild laden
        File file = new File(IMAGE_PATH);
        if (!file.exists()) {
            System.err.println("Bild nicht gefunden: " + file.getAbsolutePath());
            return;
        }
        BufferedImage originalImage = ImageIO.read(file);
        
        // GUI starten
        JFrame frame = new JFrame("K-Means Live Demo: Wie sieht der Algorithmus?");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(1, 4)); // 4 Bilder nebeneinander

        // 2. Original anzeigen
        frame.add(createPanel(originalImage, "Original"));

        // 3. K-Means mit verschiedenen k live berechnen und anzeigen
        frame.add(createPanel(runKMeansOnImage(originalImage, 2), "k = 2"));
        frame.add(createPanel(runKMeansOnImage(originalImage, 6), "k = 6"));
        frame.add(createPanel(runKMeansOnImage(originalImage, 32), "k = 32"));

        frame.pack();
        frame.setSize(1200, 400);
        frame.setVisible(true);
        frame.setLocationRelativeTo(null);
    }

    /**
     * Führt K-Means auf den Pixeln eines Bildes aus und reduziert die Farben.
     */
    public static BufferedImage runKMeansOnImage(BufferedImage img, int k) {
        int w = img.getWidth();
        int h = img.getHeight();
        
        // A) Bild in Datenmatrix umwandeln (Pixel x 3 RGB)
        double[][] data = new double[w * h][3];
        int index = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                data[index][0] = (rgb >> 16) & 0xFF; // Red
                data[index][1] = (rgb >> 8) & 0xFF;  // Green
                data[index][2] = (rgb) & 0xFF;       // Blue
                index++;
            }
        }

        // B) Smile K-Means trainieren
        System.out.println("Trainiere K-Means mit k=" + k + "...");
        CentroidClustering<double[], double[]> model = KMeans.fit(data, k, 10);

        // C) Bild rekonstruieren (Jeden Pixel durch sein Cluster-Zentrum ersetzen)
        BufferedImage newImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        index = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Vorhersage: Zu welchem Cluster gehört dieser Pixel?
                int clusterId = model.predict(data[index]);
                
                // Die Farbe des Zentroids holen
                double[] center = model.centers()[clusterId];
                
                // Zurück in RGB int wandeln
                int r = (int) center[0];
                int g = (int) center[1];
                int b = (int) center[2];
                int newRgb = (r << 16) | (g << 8) | b;
                
                newImg.setRGB(x, y, newRgb);
                index++;
            }
        }
        return newImg;
    }

    // Kleiner Helfer für die GUI
    private static JPanel createPanel(BufferedImage img, String title) {
        JPanel p = new JPanel(new BorderLayout());
        JLabel lbl = new JLabel(new ImageIcon(img.getScaledInstance(250, 250, Image.SCALE_SMOOTH)));
        JLabel txt = new JLabel(title, SwingConstants.CENTER);
        txt.setFont(new Font("Arial", Font.BOLD, 16));
        p.add(lbl, BorderLayout.CENTER);
        p.add(txt, BorderLayout.NORTH);
        return p;
    }
}