package tf

import (
	"strings"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	tensorflowpb "github.com/luk-ai/lukai/protobuf/tensorflow"
)

func RefType(dt tensorflow.DataType) tensorflow.DataType {
	name := tensorflowpb.DataType_name[int32(dt)]
	return tensorflow.DataType(tensorflowpb.DataType_value[name+"_REF"])
}

func DerefType(dt tensorflow.DataType) tensorflow.DataType {
	name := tensorflowpb.DataType_name[int32(dt)]
	return tensorflow.DataType(tensorflowpb.DataType_value[strings.TrimSuffix(name, "_REF")])
}
