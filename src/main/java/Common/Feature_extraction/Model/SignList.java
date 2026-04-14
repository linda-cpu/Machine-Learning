package Common.Feature_extraction.Model;
import java.util.ArrayList;
import java.util.List;

public class SignList {
    private List<Sign> signs = new ArrayList<>();

    public List<Sign> getSigns() {
        return signs;
    }

    public void setSigns(List<Sign> signs) {
        this.signs = signs;
    }

    public void addSign(Sign sign) {
        this.signs.add(sign);
    }

}
