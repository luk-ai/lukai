package ai.luk;

import java.util.Map;
import java.util.List;

class ModelType {

  private ai.luk.bindable.ModelType mt;

  public ModelType(String domain, String modelType, String dataDir) throws Exception {
    mt = ai.luk.bindable.Bindable.makeModelType(domain, modelType, dataDir);
  }

  public void log(Map<String, Tensor> feeds, List<String> targets) throws Exception {
    mt.log(serializeMap(feeds), serializeList(targets));
  }

  public Map<String, Tensor> run(Map<String, Tensor> feeds, List<String> fetches, List<String> targets) throws Exception {
    return unserializeMap(mt.run(serializeMap(feeds), serializeList(fetches), serializeList(targets)));
  }

  private static byte[] serializeList(List<String> list) {
    return null;
  }

  private static byte[] serializeMap(Map<String, Tensor> list) {
    return null;
  }

  private static Map<String, Tensor> unserializeMap(byte[] body) {
    return null;
  }
}
