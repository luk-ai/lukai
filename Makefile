PROTO_FILES = $(shell find protobuf/ -type f -name '*.proto')
PROTO_GO_FILES = $(patsubst protobuf/%.proto, protobuf/%.pb.go, $(PROTO_FILES))

IMPORT_PREFIX := github.com/d4l3k/pok/protobuf/

# To edit in-place without creating a backup file, GNU sed requires a bare -i,
# while BSD sed requires an empty string as the following argument.
SED_INPLACE = sed $(shell sed --version 2>&1 | grep -q GNU && echo -i || echo "-i ''")
$(call make-lazy,SED_INPLACE)

.PHONY: build
build: protobuf

.PHONY: test
test: gotest

.PHONY: gotest
gotest:
	go test -v -race ./...

.PHONY: godeps
godeps: .tensorflow
	go get -t -v ./...

.tensorflow:
	curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.3.0.tar.gz" | sudo tar -C /usr/local -xz
	sudo ldconfig
	ln -s /usr/local/lib/libtensorflow.so .
	ln -s /usr/local/lib/libtensorflow.so ./tf
	touch .tensorflow

.PHONY: protobuf
protobuf: protobuf/tensorflow $(PROTO_GO_FILES) fixuppyproto

current_dir = $(shell pwd)
tensorflow_dir = "$(current_dir)/../../tensorflow/tensorflow/"
tensorflow_protobuf_dir = "$(current_dir)/protobuf/tensorflow/"
python_proto_dir =py/libpok/proto/

.PHONY: protobuf/tensorflow
protobuf/tensorflow:
	mkdir -p $(tensorflow_protobuf_dir)
	cd $(tensorflow_dir) && cp -u tensorflow/core/protobuf/saver.proto $(tensorflow_protobuf_dir)
	cd $(tensorflow_dir) && cp -u tensorflow/core/framework/types.proto $(tensorflow_protobuf_dir)
	#cd $(tensorflow_dir) && cp -u --parents `find -name \*.proto | sed '/tensorflow\/python/d' | sed '/_service.proto/d'` $(current_dir)/protobuf/

proto_import_paths=-I ${GOPATH}/src -I ${GOPATH}/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis -I protobuf/

%.pb.go: %.proto
	protoc --proto_path=${GOPATH}/src:. ${proto_import_paths} $< --gogoslick_out=plugins=grpc:. --grpc-gateway_out=logtostderr=true:.
	python -m grpc_tools.protoc ${proto_import_paths} --python_out=${python_proto_dir} --grpc_python_out=${python_proto_dir} $<

.PHONY: fixuppyproto
fixuppyproto:
	$(SED_INPLACE) '/import gogo_pb2/d' $(shell find ${python_proto_dir} -type f -name '*.py')
	$(SED_INPLACE) 's/github_dot_com_dot_gogo_dot_protobuf_dot_gogoproto_dot_gogo__pb2.DESCRIPTOR,//g' $(shell find ${python_proto_dir} -type f -name '*.py')
	$(SED_INPLACE) '/import annotations_pb2/d' $(shell find ${python_proto_dir} -type f -name '*.py')
	$(SED_INPLACE) 's/google_dot_api_dot_annotations__pb2.DESCRIPTOR,//g' $(shell find ${python_proto_dir} -type f -name '*.py')
	$(SED_INPLACE) 's/github.com.d4l3k.pok.protobuf/libpok.proto/g' $(shell find ${python_proto_dir} -type f -name '*.py')
	$(SED_INPLACE) -E 's/^from (libpok.proto.)*/from libpok.proto./g' $(shell find ${python_proto_dir} -type f -name '*_pb2_grpc.py')


.PHONY: clean
clean:
	find . -type f -name '*.pb.go' -delete
	find . -type f -name '*_pb2_grpc.py' -delete
	find . -type f -name '*_pb2.py' -delete

.PHONY: loc
loc:
	cloc --3 $(shell find . -type f | sed '/_pb2/d' | sed '/.pb.go/d' | sed '/proto\/tensorflow/d' | sed '/.git/d' | sed '/__pycache__/d' | sed '/.egg-info/d' | sed '/testdata/d')

