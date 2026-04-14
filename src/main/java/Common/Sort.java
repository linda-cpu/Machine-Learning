package Common;

import java.io.*;
import java.util.*;

public class Sort {

    public static void main(String[] args) {
        String inputFile = "ergebnisse.csv";         // Deine Original-Datei
        String outputFile = "ergebnisse_final.csv";  // Die neue Datei mit den Extra-Spalten

        System.out.println("Bearbeite Datei...");

        try (BufferedReader br = new BufferedReader(new FileReader(inputFile));
             PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {

            String line;
            boolean isFirstLine = true;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                // 1. Header bearbeiten
                if (isFirstLine) {
                    // Wir fügen vorne zwei Spalten-Namen hinzu
                    pw.println("Json_Name;Cluster_Count;" + line);
                    isFirstLine = false;
                    continue;
                }

                // 2. Datenzeilen bearbeiten
                try {
                    String[] cols = line.split(";");
                    String modelName = cols[1]; // Das ist z.B. "K-Means-5-traffic_Alex.json"

                    // Wir holen uns die Infos aus dem Namen
                    String[] info = extractInfo(modelName);
                    String jsonName = info[0];
                    String clusterCount = info[1];

                    // 3. Neue Zeile schreiben: Neue Spalten + Alte Zeile
                    pw.println(jsonName + ";" + clusterCount + ";" + line);

                } catch (Exception e) {
                    // Falls eine Zeile kaputt ist, schreiben wir "Error" in die Spalten,
                    // damit die CSV nicht verrutscht.
                    pw.println("Error;0;" + line);
                }
            }

            System.out.println("Fertig! Neue Datei erstellt: " + outputFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Zerlegt "K-Means-5-traffic_Alex.json" in ["traffic_Alex.json", "5"]
     */
    private static String[] extractInfo(String modelName) {
        try {
            // Wir splitten am Bindestrich "-"
            String[] parts = modelName.split("-");
            
            // Format ist: K - Means - ZAHL - NAME
            // Index:      0     1      2      3...
            
            if (parts.length < 3) return new String[]{"Unbekannt", "0"};

            String k = parts[2]; // Die Zahl (z.B. "5")

            // Der Name ist der ganze Rest (falls der Dateiname auch Bindestriche hat)
            StringBuilder name = new StringBuilder();
            for (int i = 3; i < parts.length; i++) {
                name.append(parts[i]);
                if (i < parts.length - 1) name.append("-");
            }

            return new String[]{name.toString(), k};

        } catch (Exception e) {
            return new String[]{"ParseError", "0"};
        }
    }
}