# Android Client Library

This is the home of the android client library. It's a combination of Tensorflow C for Android, the Go client library, and an easy to use Java wrapper around them.

## Get Started

Download `bindable.aar` and `library-release.aar` from the [releases page](https://godoc.org/golang.org/x/mobile/cmd/gomobile) and include them in your android project.

```java
package ai.luk.example;

import ai.luk.ModelType;
import ai.luk.Tensor;
import ai.luk.DataType;

class App {
  public void train() {
    ModelType mt = new ModelType("domain", "mnist", "./mnist-files/");
    
    // Run the production model.
    feeds = new HashMap<String, Tensor>();
    feeds.put("Placeholder:0", Tensor.create(1.0f, DataType.FLOAT));
    Map<String, Tensor> outputs = mt.run(feeds, 
      Arrays.asList("Placeholder_1:0"), 
      Arrays.asList("extra inference target (optional)"));
    
    
    // Log some training examples.
    // Typically you'll pair the inputs to mt.run above with the true value and then log them. 
    // That way it creates a feedback loop to improve your model over time.
    Map<String, Tensor> feeds = new HashMap<String, Tensor>();
    feeds.put("Placeholder:0", Tensor.create(1.0f, DataType.FLOAT)); // Data type is optional
    feeds.put("Placeholder_1:0", Tensor.create(1.0f, DataType.FLOAT));
    mt.log(feeds, Arrays.asList("training_target"));


    // You will need to handle when to start and stop training. 
    // Should train when the user isn't using their phone (screen off), while charging and connected to WiFi.
    mt.startTraining();
    mt.stopTraining();
   
    // Check if there were any training or example errors.
    mt.trainingError();
    mt.examplesError();
    
    // Close the model.
    mt.close(); 
  }
}
```

## Build from Source

### Dependencies

* android-sdk (and related tools)
* android-ndk-r15c
* android-platform-21
* Bazel
* Go
* [gomobile](https://godoc.org/golang.org/x/mobile/cmd/gomobile)

To build you need to install android and build
[Tensorflow for android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android).
You'll need to use the
[luk-ai/tensorflow fork](https://github.com/luk-ai/tensorflow)
and then run the commands below. Other versions of the NDK may work, but are
untested.

ENV variables:

```
export ANDROID_HOME=/opt/android-sdk
export ANDROID_SDK=$ANDROID_HOME
export ANDROID_SDK_HOME=$ANDROID_HOME
export ANDROID_SDK_API_LEVEL=23
export ANDROID_NDK_HOME=$HOME/Developer/android-ndk-r15c
export ANDROID_NDK_API_LEVEL=23
export ANDROID_BUILD_TOOLS_VERSION=27.0.3
```

Run `./configure` to apply those envariables. Make sure to manually configure android in case it missed anything.

### Building Tensorflow for Android

#### Arm

```
sed -i 's/API_LEVEL="21"/API_LEVEL="15"/g' .tf_configure.bazelrc
bazel build --config=android_arm //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/arm
cp $ANDROID_NDK_HOME/platforms/android-15/arch-arm/usr/lib/* $GOPATH/pkg/gomobile/lib/arm/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm/libtensorflow.so
```

#### Arm64

```
sed -i 's/API_LEVEL="15"/API_LEVEL="21"/g' .tf_configure.bazelrc
bazel build --config=android_arm64 //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/arm64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-arm64/usr/lib/* $GOPATH/pkg/gomobile/lib/arm64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm64/libtensorflow.so
```

#### x86

```
sed -i 's/API_LEVEL="21"/API_LEVEL="15"/g' .tf_configure.bazelrc
bazel build --config=android //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11' --cpu=x86 --fat_apk_cpu=x86
mkdir -p $GOPATH/pkg/gomobile/lib/386
cp $ANDROID_NDK_HOME/platforms/android-15/arch-x86/usr/lib/* $GOPATH/pkg/gomobile/lib/386/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/386/libtensorflow.so
```

#### x86_64

```
sed -i 's/API_LEVEL="15"/API_LEVEL="21"/g' .tf_configure.bazelrc
bazel build --config=android //tensorflow/contrib/android:libtensorflow_inference.so --cpu=x86_64 --fat_apk_cpu=x86_64  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/amd64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-x86_64/usr/lib/* $GOPATH/pkg/gomobile/lib/amd64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/amd64/libtensorflow.so
```

### Build bindable.aar (Tensorflow + Go Client Library)

Once you've built Tensorflow for all of the various architectures you have to build the Go wrapper and package it all up into one AAR file.

```
cd android
make
```

Final AAR file will be at `android/bindable/bindable.aar`

### Build Java client library

Once you've built `bindable.aar` you need to build the Java client library.

```
# Run tests
gradle :library:test :library:connectedDebugAndroidTest --stacktrace --info
# Build
gradle :library:assemble --stacktrace --info
```

Final AAR file is found at `android/library/build/outputs/aar/library-release.aar` you need to include this and `bindable.aar` in the final application. 


### Old way to build Tensorflow For Android

Commands to build Android dependencies:

#### Arm
```
sed -i 's/api_level=21,/api_level=15,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a
mkdir -p $GOPATH/pkg/gomobile/lib/arm
cp $ANDROID_NDK_HOME/platforms/android-15/arch-arm/usr/lib/* $GOPATH/pkg/gomobile/lib/arm/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm/libtensorflow.so
```

#### Arm64
```
sed -i 's/api_level=15,/api_level=21,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=arm64-v8a
mkdir -p $GOPATH/pkg/gomobile/lib/arm64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-arm64/usr/lib/* $GOPATH/pkg/gomobile/lib/arm64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm64/libtensorflow.so
```

#### x86
```
sed -i 's/api_level=21,/api_level=15,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=x86
mkdir -p $GOPATH/pkg/gomobile/lib/386
cp $ANDROID_NDK_HOME/platforms/android-15/arch-x86/usr/lib/* $GOPATH/pkg/gomobile/lib/386/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/386/libtensorflow.so
```

#### x86_64
```
sed -i 's/api_level=15,/api_level=21,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=x86_64
mkdir -p $GOPATH/pkg/gomobile/lib/amd64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-x86_64/usr/lib/* $GOPATH/pkg/gomobile/lib/amd64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/amd64/libtensorflow.so
```
