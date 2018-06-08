.PHONY: build
build:
	gomobile bind -x -v -target=android -javapkg=ai.luk github.com/luk-ai/lukai/bindable
	-rm -rf bindable-aar/
	unzip bindable.aar -d bindable-aar
	cp $(GOPATH)/pkg/gomobile/lib/arm/libtensorflow.so bindable-aar/jni/armeabi-v7a/
	cp $(GOPATH)/pkg/gomobile/lib/arm64/libtensorflow.so bindable-aar/jni/arm64-v8a/
	cp $(GOPATH)/pkg/gomobile/lib/386/libtensorflow.so bindable-aar/jni/x86/
	cp $(GOPATH)/pkg/gomobile/lib/amd64/libtensorflow.so bindable-aar/jni/x86_64/
	jar cvf bindable.aar -C bindable-aar/ .
	cp bindable.aar bindable/
