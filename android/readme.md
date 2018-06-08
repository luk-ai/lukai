# Android Client Library

This is the home of the android client library. It's mostly generated bindings
via gomobile.

## Dependencies

* android-sdk (and related tools)
* android-ndk-r15c
* android-platform-23

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

Commands to build Android dependencies:

### Arm

```
sed -i 's/API_LEVEL="21"/API_LEVEL="15"/g' .tf_configure.bazelrc
bazel build --config=android_arm //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/arm
cp $ANDROID_NDK_HOME/platforms/android-15/arch-arm/usr/lib/* $GOPATH/pkg/gomobile/lib/arm/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm/libtensorflow.so
```

### Arm64

```
sed -i 's/API_LEVEL="15"/API_LEVEL="21"/g' .tf_configure.bazelrc
bazel build --config=android_arm64 //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/arm64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-arm64/usr/lib/* $GOPATH/pkg/gomobile/lib/arm64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/arm64/libtensorflow.so
```

### x86

```
sed -i 's/API_LEVEL="21"/API_LEVEL="15"/g' .tf_configure.bazelrc
bazel build --config=android //tensorflow/contrib/android:libtensorflow_inference.so  --cxxopt='--std=c++11' --cpu=x86 --fat_apk_cpu=x86
mkdir -p $GOPATH/pkg/gomobile/lib/386
cp $ANDROID_NDK_HOME/platforms/android-15/arch-x86/usr/lib/* $GOPATH/pkg/gomobile/lib/386/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/386/libtensorflow.so
```

### x86_64

```
sed -i 's/API_LEVEL="15"/API_LEVEL="21"/g' .tf_configure.bazelrc
bazel build --config=android //tensorflow/contrib/android:libtensorflow_inference.so --cpu=x86_64 --fat_apk_cpu=x86_64  --cxxopt='--std=c++11'
mkdir -p $GOPATH/pkg/gomobile/lib/amd64
cp $ANDROID_NDK_HOME/platforms/android-21/arch-x86_64/usr/lib/* $GOPATH/pkg/gomobile/lib/amd64/
cp -f bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $GOPATH/pkg/gomobile/lib/amd64/libtensorflow.so
```

## Old

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
