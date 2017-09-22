# Android Client Library

This is the home of the android client library. It's mostly generated bindings
via gomobile.

## Dependencies

* android-sdk (and related tools)
* android-ndk-r14b
* android-platform-23

To build you need to install android and build
[Tensorflow for android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android).
You'll need to use the
[luk-ai/tensorflow fork](https://github.com/luk-ai/tensorflow)
and then run the commands below. Other versions of the NDK may work, but are
untested.

Commands to build Android dependencies:

### Arm
```
sed -i 's/api_level=21,/api_level=15,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a
mkdir -p $GOPATH/pkg/gomobile/lib/arm
cp $ANDROID_NDK_HOME/platforms/android-15/arch-arm/usr/lib/* $GOPATH/pkg/gomobile/lib/arm/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm/libtensorflow.so
```

### Arm64
```
sed -i 's/api_level=15,/api_level=21,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=arm64-v8a
mkdir -p $GOPATH/pkg/gomobile/lib/arm64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-arm64/usr/lib/* $GOPATH/pkg/gomobile/lib/arm64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm64/libtensorflow.so
```

### x86
```
sed -i 's/api_level=21,/api_level=15,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=x86
mkdir -p $GOPATH/pkg/gomobile/lib/386
cp $ANDROID_NDK_HOME/platforms/android-15/arch-x86/usr/lib/* $GOPATH/pkg/gomobile/lib/386/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/386/libtensorflow.so
```

### x86_64
```
sed -i 's/api_level=15,/api_level=21,/g' WORKSPACE
bazel build -c opt --verbose_failures //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=x86_64
mkdir -p $GOPATH/pkg/gomobile/lib/amd64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-x86_64/usr/lib/* $GOPATH/pkg/gomobile/lib/amd64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/amd64/libtensorflow.so
```
