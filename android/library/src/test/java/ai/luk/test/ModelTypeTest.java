package ai.luk.test;

import ai.luk.ModelType;

import java.util.List;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ModelTypeTest {
  @Test
  public void serializeList() {
    ModelType.serialize(Arrays.asList("a", "b"));
    ModelType.serialize(Arrays.asList(1.0f));
    ModelType.serialize(Arrays.asList(1.0));
    ModelType.serialize(Arrays.asList((int)1));
    ModelType.serialize(Arrays.asList((long)1));
    ModelType.serialize(Arrays.asList(Arrays.asList("a")));
  }
}
