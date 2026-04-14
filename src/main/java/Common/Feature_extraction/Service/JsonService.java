package Common.Feature_extraction.Service;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import Common.VectorData;
import Common.Feature_extraction.Model.Sign;
import Common.Feature_extraction.Model.SignList;

public class JsonService {
	private String author = "Linda";
	private List<String> properties = Arrays.asList("Total_red", "Total_unknown", "Total_yellow", "Total_blue",
			"Total_light", "Total_black", "Q1_red", "Q1_unknown", "Q1_yellow", "Q1_blue", "Q1_light", "Q1_black",
			"Q2_red", "Q2_unknown", "Q2_yellow", "Q2_blue", "Q2_light", "Q2_black", "Q3_red", "Q3_unknown", "Q3_yellow",
			"Q3_blue", "Q3_light", "Q3_black", "Q4_red", "Q4_unknown", "Q4_yellow", "Q4_blue", "Q4_light", "Q4_black");

	public void prepareSignForJSON(SignList signList) {
		List<VectorData> values = new ArrayList<VectorData>();
		for (Sign sign : signList.getSigns()) {
			VectorData value = new VectorData();
			System.out.println(sign.getColorPercentageValues());
			value.setValues(sign.getColorPercentageValues());
			value.setConcept(sign.getConcept());
			values.add(value);
		}
		createJson(values);
	}

	public void createJson(List<VectorData> values) {
		try (JsonWriter writer = new JsonWriter(new FileWriter("src/main/resources/json/traffic.json"))) {
			writer.beginObject();

			writer.name("author").value(this.author);

			writer.name("properties");
			writer.beginArray();
			for (String prop : this.properties) {
				writer.value(prop);
			}
			writer.endArray();

			writer.name("vectors");
			writer.beginArray();

			for (VectorData value : values) {
				writer.beginObject();

				writer.name("concept").value(value.getConcept().toString());

				writer.name("values");
				writer.beginArray();
				if (value.getValues() != null) {
					for (float v : value.getValues()) {
						writer.value(v);
					}
				}
				writer.endArray();

				writer.endObject();
			}

			writer.endArray();
			writer.endObject();

			System.out.println("JSON created!");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public List<VectorData> setTrainingFromFile(String path) throws IOException {
		List<VectorData> dataList = new ArrayList<>();
		Gson gson = new Gson();

		try (JsonReader reader = new JsonReader(new FileReader(path))) {
			reader.beginObject();
			while (reader.hasNext()) {
				String name = reader.nextName();

				if (name.equals("vectors")) {
					reader.beginArray();

					while (reader.hasNext()) {
						VectorData vector = gson.fromJson(reader, VectorData.class);

						dataList.add(vector);
					}
					reader.endArray();
				} else {
					reader.skipValue();
				}
			}

			reader.endObject();
			System.out.println("Data loaded: " + dataList.size() + " vectors.");

		} catch (Exception e) {
			e.printStackTrace();
		}

		return dataList;
	}
}
