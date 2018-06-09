package ai.luk;

import java.util.Map;
import java.util.List;
import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.lang.String;

public class ModelType {
  public ai.luk.bindable.ModelType mt;

  public ModelType(String domain, String modelType, String dataDir) throws Exception {
    mt = ai.luk.bindable.Bindable.makeModelType(domain, modelType, dataDir);
  }

  @Override
  public void finalize() throws Throwable {
    super.finalize();

    if (mt != null) {
      mt.close();
    }
  }

  // isTraining returns whether or not the model is training.
  public boolean isTraining() {
    return mt.isTraining();
  }

  // startTraining launches an async process to wait for training jobs and
  // process them.
  public void startTraining() throws Exception {
    mt.startTraining();
  }

  // stopTraining stops the model from waiting for training jobs to finish and
  // cancels any currently running jobs.
  public void stopTraining() throws Exception {
    mt.stopTraining();
  }

  // close closes the model. This is to ensure early closure. There is a
  // finalizer that will call this method if it isn't already.
  public void close() throws Exception {
    mt.close();
  }

  // examplesError will throw an exception if there was an error during
  // saving examples.
  public void examplesError() throws Exception {
    mt.examplesError();
  }

  // trainingError will throw an exception if there was an error during
  // training.
  public void trainingError() throws Exception {
    mt.trainingError();
  }

  // totalExamples returns the total number of examples the model has.
  public long totalExamples() {
    return mt.totalExamples();
  }

  // setMaxConcurrentTrainingJobs set the maximum number of concurrent training
  // jobs that can be running. Defaults to 1.
  public void setMaxConcurrentTrainingJobs(int n) {
    mt.setMaxConcurrentTrainingJobs(n);
  }

  // log logs a training example to disk along with the target to run to train
  // with it.
  public void log(Map<String, Tensor> feeds, List<String> targets) throws Exception {
    mt.log(serialize(feeds), serialize(targets));
  }

  // run runs the production model with the specified targets and returns the
  // fetches.
  public Map<String, Tensor> run(
      Map<String, Tensor> feeds, List<String> fetches, List<String> targets) throws Exception {
    return unserializeMap(mt.run(serialize(feeds), serialize(fetches), serialize(targets)));
  }

  public static void serializeTo(ByteArrayOutputStream w, long n) {
    ByteBuffer buffer = ByteBuffer.allocate(Long.SIZE / Byte.SIZE).order(ByteOrder.nativeOrder());
    buffer.putLong(n);
    byte[] body = buffer.array();
    w.write(body, 0, body.length);
  };

  public static void serializeTo(ByteArrayOutputStream w, float n) {
    ByteBuffer buffer = ByteBuffer.allocate(Float.SIZE / Byte.SIZE).order(ByteOrder.nativeOrder());
    buffer.putFloat(n);
    byte[] body = buffer.array();
    w.write(body, 0, body.length);
  };

  public static void serializeTo(ByteArrayOutputStream w, double n) {
    ByteBuffer buffer = ByteBuffer.allocate(Double.SIZE / Byte.SIZE).order(ByteOrder.nativeOrder());
    buffer.putDouble(n);
    byte[] body = buffer.array();
    w.write(body, 0, body.length);
  };

  public static void serializeTo(ByteArrayOutputStream w, int n) {
    ByteBuffer buffer = ByteBuffer.allocate(Integer.SIZE / Byte.SIZE).order(ByteOrder.nativeOrder());
    buffer.putInt(n);
    byte[] body = buffer.array();
    w.write(body, 0, body.length);
  };

  public static void serializeTo(ByteArrayOutputStream w, Integer n) {
    serializeTo(w, (int)n);
  };

  public static void serializeTo(ByteArrayOutputStream w, Float n) {
    serializeTo(w, (float)n);
  };

  public static void serializeTo(ByteArrayOutputStream w, Long n) {
    serializeTo(w, (long)n);
  };

  public static void serializeTo(ByteArrayOutputStream w, Double n) {
    serializeTo(w, (double)n);
  };

  public static void serializeTo(ByteArrayOutputStream w, long[] arr) {
    serializeTo(w, (long) arr.length);
    for (long item : arr) {
      serializeTo(w, item);
    }
  };

  public static void serializeTo(ByteArrayOutputStream w, String str) {
    try {
      byte[] body = str.getBytes("UTF-8");
      serializeTo(w, (long) body.length);
      w.write(body, 0, body.length);
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException(e);
    }
  }

  public static void serializeTo(ByteArrayOutputStream w, Tensor t) {
    serializeTo(w, (long) t.dataType().c());
    serializeTo(w, t.shape());
    byte[] body = t.bytesValue();
    serializeTo(w, (long) body.length);
    w.write(body, 0, body.length);
  }

  public static <T> void serializeTo(ByteArrayOutputStream w, List<T> list) {
    serializeTo(w, (long) list.size());
    for (T item : list) {
      serializeTo(w, item);
    }
  }

  public static void serializeTo(ByteArrayOutputStream w, Object unknown) {
    if (unknown instanceof String) {
      serializeTo(w, (String)unknown);
    } else if (unknown instanceof List) {
      serializeTo(w, (List)unknown);
    } else if (unknown instanceof Integer) {
      serializeTo(w, (Integer)unknown);
    } else if (unknown instanceof Long) {
      serializeTo(w, (Long)unknown);
    } else if (unknown instanceof Float) {
      serializeTo(w, (Float)unknown);
    } else if (unknown instanceof Double) {
      serializeTo(w, (Double)unknown);
    } else {
      throw new RuntimeException("unknown type serialize: " + unknown.getClass().getName());
    }
  }

  public static byte[] serialize(List<?> list) {
    ByteArrayOutputStream w = new ByteArrayOutputStream();
    serializeTo(w, list);
    return w.toByteArray();
  }

  public static byte[] serialize(Map<String, Tensor> list) throws UnsupportedEncodingException {
    ByteArrayOutputStream w = new ByteArrayOutputStream();
    serializeTo(w, (long) list.size());
    for (Map.Entry<String, Tensor> entry : list.entrySet()) {
      serializeTo(w, entry.getKey());
      serializeTo(w, entry.getValue());
    }
    return w.toByteArray();
  }

  public static Map<String, Tensor> unserializeMap(byte[] body) {
    return null;
  }
}
