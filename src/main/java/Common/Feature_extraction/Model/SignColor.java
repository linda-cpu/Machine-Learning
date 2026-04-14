package Common.Feature_extraction.Model;
public class SignColor {
    public int r;
    public int g;
    public int b;

    public SignColor(int r, int g, int b) {
		this.r = r;
		this.g = g;
		this.b = b;
	}

    public ColorCategory getColorBright() {
        if (Math.abs(r - g) < 10 && Math.abs(g - b) < 10 && Math.abs(r - b) < 10 && r > 245) {
            return ColorCategory.LIGHT;
        }
        if ((r > g + 20 && r > b + 20) && (Math.abs(g - b) < 40)) {
           return ColorCategory.RED;
        } else if ((r >= g && r > b) && (g - b > 40)) {
            return ColorCategory.YELLOW;
        } else if (Math.abs(r - g) < 30 && Math.abs(g - b) < 30 && Math.abs(r - b) < 30) {
            return ColorCategory.BLACK;
        } else if (r < g && g < b && r > 170) {
            return ColorCategory.BLUE;
        }

        return ColorCategory.UNKNOWN;
    }

    public ColorCategory getColorDark() {
        if ((r > g + 20 && r > b + 20) && (Math.abs(g - b) < 40)) {
            return ColorCategory.RED;
        } else if ((r >= g && r > b) && (g - b > 40)) {
            return ColorCategory.YELLOW;
        } else if (Math.abs(r - g) < 15 && Math.abs(g - b) < 15 && Math.abs(r - b) < 15 && r < 30) {
            return ColorCategory.BLACK;
        } else if (r < g && g < b && r < 75) {
            return ColorCategory.BLUE;
        }

        return ColorCategory.UNKNOWN;
    }
}
