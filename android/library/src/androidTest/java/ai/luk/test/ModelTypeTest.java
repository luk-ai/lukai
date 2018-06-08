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
