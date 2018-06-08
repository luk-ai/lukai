package ai.luk.test;

import ai.luk.ModelType;
import ai.luk.Tensor;
import ai.luk.DataType;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import android.support.test.runner.AndroidJUnit4;

@RunWith(AndroidJUnit4.class)
public class ModelTypeTest {

  @Rule
  public TemporaryFolder tempdir = new TemporaryFolder();

  @Test
  public void log() throws Exception {
    File dir = tempdir.newFolder("lukai-java-test");
    ModelType mt = new ModelType("test", "test", dir.getPath());
    mt.close();
  }

  @Test
  public void logTypes() throws Exception {
    File dir = tempdir.newFolder("lukai-java-test");
    ModelType mt = new ModelType("test", "test", dir.getPath());

    List<Tensor> tensors = Arrays.asList(
      Tensor.create(1.0f, DataType.FLOAT),
      Tensor.create(new float[]{1.0f}, DataType.FLOAT),
      Tensor.create(1.0, DataType.DOUBLE),
      Tensor.create(new double[]{1.0}, DataType.DOUBLE),
      Tensor.create((int)1, DataType.INT32),
      Tensor.create(new int[]{1}, DataType.INT32),
      Tensor.create((long)1, DataType.INT64),
      Tensor.create(new long[]{1}, DataType.INT64),
      Tensor.create(true, DataType.BOOL),
      Tensor.create(new boolean[]{true}, DataType.BOOL),
      Tensor.create((byte)1, DataType.UINT8),
      Tensor.create(new byte[]{1}, DataType.UINT8));

    // TODO: support DataType.STRING
    // Can work around by passing a byte array in most cases.

    for (Tensor t : tensors) {
      Map<String, Tensor> feeds = new HashMap<String, Tensor>();
      feeds.put("a:0", t);
      mt.log(feeds, Arrays.asList("target"));
    }

    mt.close();
  }

  @Test
  public void training() throws Exception {
    File dir = tempdir.newFolder("lukai-java-test");
    ModelType mt = new ModelType("test", "test", dir.getPath());

    Map<String, Tensor> feeds = new HashMap<String, Tensor>();
    feeds.put("Placeholder:0", Tensor.create(1.0f, DataType.FLOAT));
    feeds.put("Placeholder_1:0", Tensor.create(1.0f, DataType.FLOAT));
    mt.log(feeds, Arrays.asList("adam_optimizer/Adam"));
    assertEquals(mt.totalExamples(), 1);
    mt.startTraining();
    assertTrue(mt.isTraining());
    mt.stopTraining();

    mt.trainingError();
    mt.examplesError();

    mt.close();
  }
}
