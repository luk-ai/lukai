package ai.luk;

import java.util.Map;
import java.util.List;
import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ModelType {
  private ai.luk.bindable.ModelType mt;

  public ModelType(String domain, String modelType, String dataDir) throws Exception {
    mt = ai.luk.bindable.Bindable.makeModelType(domain, modelType, dataDir);
  }

  public boolean isTraining() {
    return mt.isTraining();
  }

  public void startTraining() throws Exception {
    mt.startTraining();
  }

  public void stopTraining() throws Exception {
    mt.stopTraining();
  }

  public long totalExamples() {
    return mt.totalExamples();
  }

  public void log(Map<String, Tensor> feeds, List<String> targets) throws Exception {
    mt.log(serialize(feeds), serialize(targets));
  }

  public Map<String, Tensor> run(
      Map<String, Tensor> feeds, List<String> fetches, List<String> targets) throws Exception {
    return unserializeMap(mt.run(serialize(feeds), serialize(fetches), serialize(targets)));
  }

  private static void serializeTo(ByteArrayOutputStream w, long n) {
    ByteBuffer buffer = ByteBuffer.allocate(Long.SIZE / Byte.SIZE).order(ByteOrder.nativeOrder());
    buffer.putLong(n);
    byte[] body = buffer.array();
    w.write(body, 0, body.length);
  };

  private static void serializeTo(ByteArrayOutputStream w, String str)
      throws UnsupportedEncodingException {
    byte[] body = str.getBytes("UTF-8");
    serializeTo(w, (long) body.length);
    w.write(body, 0, body.length);
  }

  private static void serializeTo(ByteArrayOutputStream w, Tensor t) {
    serializeTo(w, (long) t.dataType().c());
    serializeTo(w, t.shape());
    byte[] body = t.bytesValue();
    serializeTo(w, (long) body.length);
    w.write(body, 0, body.length);
  }

  private static <T> void serializeTo(ByteArrayOutputStream w, List<T> list) {
    serializeTo(w, (long) list.size());
    for (T item : list) {
      serializeTo(w, item);
    }
  }

  private static void serializeTo(ByteArrayOutputStream w, Object unknown) {
    throw new RuntimeException("unknown type!");
  }

  private static <T> byte[] serialize(List<T> list) {
    ByteArrayOutputStream w = new ByteArrayOutputStream();
    serializeTo(w, list);
    return w.toByteArray();
  }

  private static byte[] serialize(Map<String, Tensor> list) throws UnsupportedEncodingException {
    ByteArrayOutputStream w = new ByteArrayOutputStream();
    serializeTo(w, (long) list.size());
    for (Map.Entry<String, Tensor> entry : list.entrySet()) {
      serializeTo(w, entry.getKey());
      serializeTo(w, entry.getValue());
    }
    return w.toByteArray();
  }

  private static Map<String, Tensor> unserializeMap(byte[] body) {
    return null;
  }
}
