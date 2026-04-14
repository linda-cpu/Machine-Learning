package Common.Feature_extraction.Service;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import Common.Feature_extraction.Model.ColorCategory;
import Common.Feature_extraction.Model.SignColor;

public class ImageService {

    public BufferedImage ImageLoader(String path) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(path));
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
        return img;
    }

    public static BufferedImage resize(BufferedImage img, int targetWidth, int targetHeight) {
        int originalWidth = img.getWidth();
        int originalHeight = img.getHeight();

        float scaleX = (float) targetWidth / originalWidth;
        float scaleY = (float) targetHeight / originalHeight;
        float scale = Math.min(scaleX, scaleY);

        int scaledWidth = (int) (originalWidth * scale);
        int scaledHeight = (int) (originalHeight * scale);

        int x = (targetWidth - scaledWidth) / 2;
        int y = (targetHeight - scaledHeight) / 2;

        BufferedImage paddedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = paddedImage.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, targetWidth, targetHeight);

        g.drawImage(img, x, y, scaledWidth, scaledHeight, null);
        g.dispose();

        return paddedImage;
    }

    public BufferedImage ImageCropped(BufferedImage img) {
        int width = img.getWidth();

        width = (width * 60) / img.getHeight();

        img = resize(img, width, 60);

        int height = img.getHeight();

        int minX = width, minY = height;
        int maxX = 0, maxY = 0;

        int threshold = 25; // Threshold: 10-20 fine, 25-40 coarse, 50+ only very strong contrasts

        for (int y = 1; y < height; y++) {
            for (int x = 1; x < width; x++) {
                Color c1 = new Color(img.getRGB(x, y));
                Color c2 = new Color(img.getRGB(x - 1, y - 1));

                int diff = colorDifference(c1, c2);
                if (diff > threshold) {
                    if (x < minX)
                        minX = x;
                    if (y < minY)
                        minY = y;
                    if (x > maxX)
                        maxX = x;
                    if (y > maxY)
                        maxY = y;
                }
            }
        }
        if (minX > maxX && minY > maxY) {
            System.out.println("No significant color differences found.");
            return img;
        }

        int croppedWidth = maxX - minX + 1;
        int croppedHeight = maxY - minY + 1;
        int size = Math.max(croppedWidth, croppedHeight);
        BufferedImage squareImage = resize(img, size, size);

        return squareImage;
    }

    public static int colorDifference(Color c1, Color c2) {
        int dR = Math.abs(c1.getRed() - c2.getRed());
        int gR = Math.abs(c1.getGreen() - c2.getGreen());
        int dB = Math.abs(c1.getBlue() - c2.getBlue());
        return (dR + gR + dB) / 3;
    }

    public Map<String, Float> calculateColorPercentages(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int totalPixels = width * height;

        int redCount = 0, yellowCount = 0, blueCount = 0, blackCount = 0, unknownCount = 0; 

        int[][] quadrantCounts = new int[4][5];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                ColorCategory colorCategory = getColorOfPixel(argb);

                switch (colorCategory) {
                    case RED -> redCount++;
                    case YELLOW -> yellowCount++;
                    case BLUE -> blueCount++;
                    case BLACK -> blackCount++;
                    default -> unknownCount++;
                }

                int quadrant;
                if (x < halfWidth && y < halfHeight) {
                    quadrant = 0; // top left
                } else if (x >= halfWidth && y < halfHeight) {
                    quadrant = 1; // top right
                } else if (x < halfWidth && y >= halfHeight) {
                    quadrant = 2; // bottom left
                } else {
                    quadrant = 3; // bottom right
                }
                
                switch (colorCategory) {
                    case RED -> quadrantCounts[quadrant][0]++;
                    case YELLOW -> quadrantCounts[quadrant][1]++;
                    case BLUE -> quadrantCounts[quadrant][2]++;
                    case BLACK -> quadrantCounts[quadrant][3]++;
                    default -> quadrantCounts[quadrant][4]++;
                }
            }
        }

        Map<String, Float> colorPercentages = new LinkedHashMap<>();

        // Total shares
        colorPercentages.put("Total_red", ((float) redCount / totalPixels));
        colorPercentages.put("Total_yellow", ((float)yellowCount / totalPixels));
        colorPercentages.put("Total_blue", ((float)blueCount / totalPixels));
        colorPercentages.put("Total_black", ((float)blackCount / totalPixels));
        //colorPercentages.put("Total_unknown", ((float)unknownCount / totalPixels));

        // Quadrant shares
        int pixelsPerQuadrant = halfWidth * halfHeight;
        String[] quadrantNames = { "Q1", "Q2", "Q3", "Q4" };
        String[] colorNames = { "red", "yellow", "blue", "black"/*, "light", "unknown" */};

        for (int q = 0; q < 4; q++) {
            for (int c = 0; c < 4; c++) {
                float percentage = ((float) quadrantCounts[q][c] / pixelsPerQuadrant);
                colorPercentages.put(quadrantNames[q] + "_" + colorNames[c], percentage);
            }
        }

        return colorPercentages;
    }

    private ColorCategory getColorOfPixel(int argb) {
        // int alpha = (argb >> 24) & 0xff;
        int red = (argb >> 16) & 0xff;
        int green = (argb >> 8) & 0xff;
        int blue = (argb) & 0xff;
        SignColor c = new SignColor(red, green, blue);
        float luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255;
        if (luminance >= 0.5f) {
            // bright color
            return c.getColorBright();
        } else {
            // dark color
            return c.getColorDark();
        }
    }

    public double[] imageToVector(BufferedImage img, int targetWidth, int targetHeight) {
        BufferedImage resized = resize(img, targetWidth, targetHeight);
        double[] vector = new double[targetWidth * targetHeight * 3];
        int index = 0;

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resized.getRGB(x, y);

                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = (rgb) & 0xFF;

                vector[index++] = red / 255.0;
                vector[index++] = green / 255.0;
                vector[index++] = blue / 255.0;
            }
        }
        return vector;
    }
}
