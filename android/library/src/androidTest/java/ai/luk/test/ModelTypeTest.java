package ai.luk.test;

import ai.luk.ModelType;
import ai.luk.Tensor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.rules.TemporaryFolder;

import java.io.File;

import android.support.test.runner.AndroidJUnit4;

@RunWith(AndroidJUnit4.class)
public class ModelTypeTest {

  @Rule
  public TemporaryFolder tempdir = new TemporaryFolder();

  @Test
  public void log() throws Exception {
    File dir = tempdir.newFolder("lukai-java-test");
    ModelType mt = ModelType.create("test", "test", dir.getPath());
  }
}
