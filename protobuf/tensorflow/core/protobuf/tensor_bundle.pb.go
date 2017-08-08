// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: protobuf/tensorflow/core/protobuf/tensor_bundle.proto

/*
	Package tensorflow is a generated protocol buffer package.

	It is generated from these files:
		protobuf/tensorflow/core/protobuf/tensor_bundle.proto

	It has these top-level messages:
		BundleHeaderProto
		BundleEntryProto
*/
package tensorflow

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"
import tensorflow1 "github.com/d4l3k/pok/protobuf/tensorflow/core/framework"
import tensorflow2 "github.com/d4l3k/pok/protobuf/tensorflow/core/framework"
import tensorflow3 "github.com/d4l3k/pok/protobuf/tensorflow/core/framework"
import tensorflow4 "github.com/d4l3k/pok/protobuf/tensorflow/core/framework"

import strconv "strconv"

import fmt "fmt"
import strings "strings"
import reflect "reflect"

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

// An enum indicating the endianness of the platform that produced this
// bundle.  A bundle can only be read by a platform with matching endianness.
// Defaults to LITTLE, as most modern platforms are little-endian.
//
// Affects the binary tensor data bytes only, not the metadata in protobufs.
type BundleHeaderProto_Endianness int32

const (
	LITTLE BundleHeaderProto_Endianness = 0
	BIG    BundleHeaderProto_Endianness = 1
)

var BundleHeaderProto_Endianness_name = map[int32]string{
	0: "LITTLE",
	1: "BIG",
}
var BundleHeaderProto_Endianness_value = map[string]int32{
	"LITTLE": 0,
	"BIG":    1,
}

func (BundleHeaderProto_Endianness) EnumDescriptor() ([]byte, []int) {
	return fileDescriptorTensorBundle, []int{0, 0}
}

// Special header that is associated with a bundle.
//
// TODO(zongheng,zhifengc): maybe in the future, we can add information about
// which binary produced this checkpoint, timestamp, etc. Sometime, these can be
// valuable debugging information. And if needed, these can be used as defensive
// information ensuring reader (binary version) of the checkpoint and the writer
// (binary version) must match within certain range, etc.
type BundleHeaderProto struct {
	// Number of data files in the bundle.
	NumShards  int32                        `protobuf:"varint,1,opt,name=num_shards,json=numShards,proto3" json:"num_shards,omitempty"`
	Endianness BundleHeaderProto_Endianness `protobuf:"varint,2,opt,name=endianness,proto3,enum=tensorflow.BundleHeaderProto_Endianness" json:"endianness,omitempty"`
	// Versioning of the tensor bundle format.
	Version *tensorflow4.VersionDef `protobuf:"bytes,3,opt,name=version" json:"version,omitempty"`
}

func (m *BundleHeaderProto) Reset()                    { *m = BundleHeaderProto{} }
func (*BundleHeaderProto) ProtoMessage()               {}
func (*BundleHeaderProto) Descriptor() ([]byte, []int) { return fileDescriptorTensorBundle, []int{0} }

func (m *BundleHeaderProto) GetNumShards() int32 {
	if m != nil {
		return m.NumShards
	}
	return 0
}

func (m *BundleHeaderProto) GetEndianness() BundleHeaderProto_Endianness {
	if m != nil {
		return m.Endianness
	}
	return LITTLE
}

func (m *BundleHeaderProto) GetVersion() *tensorflow4.VersionDef {
	if m != nil {
		return m.Version
	}
	return nil
}

// Describes the metadata related to a checkpointed tensor.
type BundleEntryProto struct {
	// The tensor dtype and shape.
	Dtype tensorflow3.DataType          `protobuf:"varint,1,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
	Shape *tensorflow1.TensorShapeProto `protobuf:"bytes,2,opt,name=shape" json:"shape,omitempty"`
	// The binary content of the tensor lies in:
	//   File "shard_id": bytes [offset, offset + size).
	ShardId int32 `protobuf:"varint,3,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	Offset  int64 `protobuf:"varint,4,opt,name=offset,proto3" json:"offset,omitempty"`
	Size_   int64 `protobuf:"varint,5,opt,name=size,proto3" json:"size,omitempty"`
	// The CRC32C checksum of the tensor bytes.
	Crc32C uint32 `protobuf:"fixed32,6,opt,name=crc32c,proto3" json:"crc32c,omitempty"`
	// Iff present, this entry represents a partitioned tensor.  The previous
	// fields are interpreted as follows:
	//
	//   "dtype", "shape": describe the full tensor.
	//   "shard_id", "offset", "size", "crc32c": all IGNORED.
	//      These information for each slice can be looked up in their own
	//      BundleEntryProto, keyed by each "slice_name".
	Slices []*tensorflow2.TensorSliceProto `protobuf:"bytes,7,rep,name=slices" json:"slices,omitempty"`
}

func (m *BundleEntryProto) Reset()                    { *m = BundleEntryProto{} }
func (*BundleEntryProto) ProtoMessage()               {}
func (*BundleEntryProto) Descriptor() ([]byte, []int) { return fileDescriptorTensorBundle, []int{1} }

func (m *BundleEntryProto) GetDtype() tensorflow3.DataType {
	if m != nil {
		return m.Dtype
	}
	return tensorflow3.DT_INVALID
}

func (m *BundleEntryProto) GetShape() *tensorflow1.TensorShapeProto {
	if m != nil {
		return m.Shape
	}
	return nil
}

func (m *BundleEntryProto) GetShardId() int32 {
	if m != nil {
		return m.ShardId
	}
	return 0
}

func (m *BundleEntryProto) GetOffset() int64 {
	if m != nil {
		return m.Offset
	}
	return 0
}

func (m *BundleEntryProto) GetSize_() int64 {
	if m != nil {
		return m.Size_
	}
	return 0
}

func (m *BundleEntryProto) GetCrc32C() uint32 {
	if m != nil {
		return m.Crc32C
	}
	return 0
}

func (m *BundleEntryProto) GetSlices() []*tensorflow2.TensorSliceProto {
	if m != nil {
		return m.Slices
	}
	return nil
}

func init() {
	proto.RegisterType((*BundleHeaderProto)(nil), "tensorflow.BundleHeaderProto")
	proto.RegisterType((*BundleEntryProto)(nil), "tensorflow.BundleEntryProto")
	proto.RegisterEnum("tensorflow.BundleHeaderProto_Endianness", BundleHeaderProto_Endianness_name, BundleHeaderProto_Endianness_value)
}
func (x BundleHeaderProto_Endianness) String() string {
	s, ok := BundleHeaderProto_Endianness_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *BundleHeaderProto) Equal(that interface{}) bool {
	if that == nil {
		if this == nil {
			return true
		}
		return false
	}

	that1, ok := that.(*BundleHeaderProto)
	if !ok {
		that2, ok := that.(BundleHeaderProto)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		if this == nil {
			return true
		}
		return false
	} else if this == nil {
		return false
	}
	if this.NumShards != that1.NumShards {
		return false
	}
	if this.Endianness != that1.Endianness {
		return false
	}
	if !this.Version.Equal(that1.Version) {
		return false
	}
	return true
}
func (this *BundleEntryProto) Equal(that interface{}) bool {
	if that == nil {
		if this == nil {
			return true
		}
		return false
	}

	that1, ok := that.(*BundleEntryProto)
	if !ok {
		that2, ok := that.(BundleEntryProto)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		if this == nil {
			return true
		}
		return false
	} else if this == nil {
		return false
	}
	if this.Dtype != that1.Dtype {
		return false
	}
	if !this.Shape.Equal(that1.Shape) {
		return false
	}
	if this.ShardId != that1.ShardId {
		return false
	}
	if this.Offset != that1.Offset {
		return false
	}
	if this.Size_ != that1.Size_ {
		return false
	}
	if this.Crc32C != that1.Crc32C {
		return false
	}
	if len(this.Slices) != len(that1.Slices) {
		return false
	}
	for i := range this.Slices {
		if !this.Slices[i].Equal(that1.Slices[i]) {
			return false
		}
	}
	return true
}
func (this *BundleHeaderProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&tensorflow.BundleHeaderProto{")
	s = append(s, "NumShards: "+fmt.Sprintf("%#v", this.NumShards)+",\n")
	s = append(s, "Endianness: "+fmt.Sprintf("%#v", this.Endianness)+",\n")
	if this.Version != nil {
		s = append(s, "Version: "+fmt.Sprintf("%#v", this.Version)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *BundleEntryProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 11)
	s = append(s, "&tensorflow.BundleEntryProto{")
	s = append(s, "Dtype: "+fmt.Sprintf("%#v", this.Dtype)+",\n")
	if this.Shape != nil {
		s = append(s, "Shape: "+fmt.Sprintf("%#v", this.Shape)+",\n")
	}
	s = append(s, "ShardId: "+fmt.Sprintf("%#v", this.ShardId)+",\n")
	s = append(s, "Offset: "+fmt.Sprintf("%#v", this.Offset)+",\n")
	s = append(s, "Size_: "+fmt.Sprintf("%#v", this.Size_)+",\n")
	s = append(s, "Crc32C: "+fmt.Sprintf("%#v", this.Crc32C)+",\n")
	if this.Slices != nil {
		s = append(s, "Slices: "+fmt.Sprintf("%#v", this.Slices)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTensorBundle(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *BundleHeaderProto) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *BundleHeaderProto) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.NumShards != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.NumShards))
	}
	if m.Endianness != 0 {
		dAtA[i] = 0x10
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Endianness))
	}
	if m.Version != nil {
		dAtA[i] = 0x1a
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Version.Size()))
		n1, err := m.Version.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n1
	}
	return i, nil
}

func (m *BundleEntryProto) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *BundleEntryProto) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.Dtype != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Dtype))
	}
	if m.Shape != nil {
		dAtA[i] = 0x12
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Shape.Size()))
		n2, err := m.Shape.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n2
	}
	if m.ShardId != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.ShardId))
	}
	if m.Offset != 0 {
		dAtA[i] = 0x20
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Offset))
	}
	if m.Size_ != 0 {
		dAtA[i] = 0x28
		i++
		i = encodeVarintTensorBundle(dAtA, i, uint64(m.Size_))
	}
	if m.Crc32C != 0 {
		dAtA[i] = 0x35
		i++
		i = encodeFixed32TensorBundle(dAtA, i, uint32(m.Crc32C))
	}
	if len(m.Slices) > 0 {
		for _, msg := range m.Slices {
			dAtA[i] = 0x3a
			i++
			i = encodeVarintTensorBundle(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func encodeFixed64TensorBundle(dAtA []byte, offset int, v uint64) int {
	dAtA[offset] = uint8(v)
	dAtA[offset+1] = uint8(v >> 8)
	dAtA[offset+2] = uint8(v >> 16)
	dAtA[offset+3] = uint8(v >> 24)
	dAtA[offset+4] = uint8(v >> 32)
	dAtA[offset+5] = uint8(v >> 40)
	dAtA[offset+6] = uint8(v >> 48)
	dAtA[offset+7] = uint8(v >> 56)
	return offset + 8
}
func encodeFixed32TensorBundle(dAtA []byte, offset int, v uint32) int {
	dAtA[offset] = uint8(v)
	dAtA[offset+1] = uint8(v >> 8)
	dAtA[offset+2] = uint8(v >> 16)
	dAtA[offset+3] = uint8(v >> 24)
	return offset + 4
}
func encodeVarintTensorBundle(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *BundleHeaderProto) Size() (n int) {
	var l int
	_ = l
	if m.NumShards != 0 {
		n += 1 + sovTensorBundle(uint64(m.NumShards))
	}
	if m.Endianness != 0 {
		n += 1 + sovTensorBundle(uint64(m.Endianness))
	}
	if m.Version != nil {
		l = m.Version.Size()
		n += 1 + l + sovTensorBundle(uint64(l))
	}
	return n
}

func (m *BundleEntryProto) Size() (n int) {
	var l int
	_ = l
	if m.Dtype != 0 {
		n += 1 + sovTensorBundle(uint64(m.Dtype))
	}
	if m.Shape != nil {
		l = m.Shape.Size()
		n += 1 + l + sovTensorBundle(uint64(l))
	}
	if m.ShardId != 0 {
		n += 1 + sovTensorBundle(uint64(m.ShardId))
	}
	if m.Offset != 0 {
		n += 1 + sovTensorBundle(uint64(m.Offset))
	}
	if m.Size_ != 0 {
		n += 1 + sovTensorBundle(uint64(m.Size_))
	}
	if m.Crc32C != 0 {
		n += 5
	}
	if len(m.Slices) > 0 {
		for _, e := range m.Slices {
			l = e.Size()
			n += 1 + l + sovTensorBundle(uint64(l))
		}
	}
	return n
}

func sovTensorBundle(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozTensorBundle(x uint64) (n int) {
	return sovTensorBundle(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *BundleHeaderProto) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&BundleHeaderProto{`,
		`NumShards:` + fmt.Sprintf("%v", this.NumShards) + `,`,
		`Endianness:` + fmt.Sprintf("%v", this.Endianness) + `,`,
		`Version:` + strings.Replace(fmt.Sprintf("%v", this.Version), "VersionDef", "tensorflow4.VersionDef", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *BundleEntryProto) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&BundleEntryProto{`,
		`Dtype:` + fmt.Sprintf("%v", this.Dtype) + `,`,
		`Shape:` + strings.Replace(fmt.Sprintf("%v", this.Shape), "TensorShapeProto", "tensorflow1.TensorShapeProto", 1) + `,`,
		`ShardId:` + fmt.Sprintf("%v", this.ShardId) + `,`,
		`Offset:` + fmt.Sprintf("%v", this.Offset) + `,`,
		`Size_:` + fmt.Sprintf("%v", this.Size_) + `,`,
		`Crc32C:` + fmt.Sprintf("%v", this.Crc32C) + `,`,
		`Slices:` + strings.Replace(fmt.Sprintf("%v", this.Slices), "TensorSliceProto", "tensorflow2.TensorSliceProto", 1) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTensorBundle(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *BundleHeaderProto) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorBundle
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: BundleHeaderProto: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: BundleHeaderProto: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumShards", wireType)
			}
			m.NumShards = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NumShards |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Endianness", wireType)
			}
			m.Endianness = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Endianness |= (BundleHeaderProto_Endianness(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Version", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTensorBundle
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Version == nil {
				m.Version = &tensorflow4.VersionDef{}
			}
			if err := m.Version.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTensorBundle(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTensorBundle
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *BundleEntryProto) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorBundle
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: BundleEntryProto: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: BundleEntryProto: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Dtype", wireType)
			}
			m.Dtype = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Dtype |= (tensorflow3.DataType(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Shape", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTensorBundle
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Shape == nil {
				m.Shape = &tensorflow1.TensorShapeProto{}
			}
			if err := m.Shape.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ShardId", wireType)
			}
			m.ShardId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.ShardId |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Offset", wireType)
			}
			m.Offset = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Offset |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Size_", wireType)
			}
			m.Size_ = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Size_ |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 6:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Crc32C", wireType)
			}
			m.Crc32C = 0
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += 4
			m.Crc32C = uint32(dAtA[iNdEx-4])
			m.Crc32C |= uint32(dAtA[iNdEx-3]) << 8
			m.Crc32C |= uint32(dAtA[iNdEx-2]) << 16
			m.Crc32C |= uint32(dAtA[iNdEx-1]) << 24
		case 7:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Slices", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTensorBundle
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Slices = append(m.Slices, &tensorflow2.TensorSliceProto{})
			if err := m.Slices[len(m.Slices)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTensorBundle(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTensorBundle
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipTensorBundle(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTensorBundle
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
			return iNdEx, nil
		case 1:
			iNdEx += 8
			return iNdEx, nil
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowTensorBundle
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			iNdEx += length
			if length < 0 {
				return 0, ErrInvalidLengthTensorBundle
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowTensorBundle
					}
					if iNdEx >= l {
						return 0, io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					innerWire |= (uint64(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				innerWireType := int(innerWire & 0x7)
				if innerWireType == 4 {
					break
				}
				next, err := skipTensorBundle(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
			}
			return iNdEx, nil
		case 4:
			return iNdEx, nil
		case 5:
			iNdEx += 4
			return iNdEx, nil
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
	}
	panic("unreachable")
}

var (
	ErrInvalidLengthTensorBundle = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTensorBundle   = fmt.Errorf("proto: integer overflow")
)

func init() {
	proto.RegisterFile("protobuf/tensorflow/core/protobuf/tensor_bundle.proto", fileDescriptorTensorBundle)
}

var fileDescriptorTensorBundle = []byte{
	// 461 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x92, 0xc1, 0x6e, 0xd3, 0x40,
	0x10, 0x86, 0x3d, 0x4d, 0x6d, 0xc3, 0x54, 0xaa, 0xc2, 0x82, 0x2a, 0x83, 0x60, 0x65, 0x22, 0x21,
	0x59, 0x08, 0x25, 0xc8, 0x85, 0x17, 0x88, 0x1a, 0xd1, 0x48, 0x3d, 0x54, 0x6e, 0xc4, 0x81, 0x4b,
	0xe5, 0xd8, 0x6b, 0xb0, 0x48, 0xbc, 0xd1, 0xae, 0x4d, 0x15, 0x4e, 0xdc, 0xb8, 0xf2, 0x18, 0x3c,
	0x0a, 0xc7, 0x72, 0xe3, 0x48, 0xcc, 0x85, 0x63, 0x8f, 0x1c, 0x91, 0xc7, 0x0e, 0xb1, 0x8a, 0x82,
	0xb8, 0xed, 0xcc, 0x7c, 0xf3, 0xef, 0xfc, 0xb3, 0x8b, 0xcf, 0x17, 0x4a, 0xe6, 0x72, 0x5a, 0x24,
	0x83, 0x5c, 0x64, 0x5a, 0xaa, 0x64, 0x26, 0x2f, 0x06, 0x91, 0x54, 0x62, 0x70, 0xad, 0x70, 0x3e,
	0x2d, 0xb2, 0x78, 0x26, 0xfa, 0x94, 0x66, 0xb8, 0xa1, 0xef, 0x3d, 0xb9, 0xde, 0x99, 0xa8, 0x70,
	0x2e, 0x2e, 0xa4, 0x7a, 0xbb, 0x6e, 0xd5, 0x6f, 0xc2, 0x45, 0xd3, 0xf9, 0x3f, 0xf4, 0x2c, 0x8d,
	0xd6, 0xf4, 0xa3, 0x7f, 0xd0, 0xcb, 0x85, 0xd0, 0x0d, 0xe6, 0x6d, 0xc7, 0xde, 0x09, 0xa5, 0x53,
	0x99, 0x35, 0x64, 0xef, 0x2b, 0xe0, 0xad, 0x21, 0x39, 0x39, 0x16, 0x61, 0x2c, 0xd4, 0x29, 0xd9,
	0x79, 0x80, 0x98, 0x15, 0xf3, 0x6a, 0x4e, 0x15, 0x6b, 0x07, 0x5c, 0xf0, 0xcc, 0xe0, 0x66, 0x56,
	0xcc, 0xcf, 0x28, 0xc1, 0x8e, 0x11, 0x45, 0x16, 0xa7, 0x61, 0x96, 0x09, 0xad, 0x9d, 0x1d, 0x17,
	0xbc, 0x7d, 0xdf, 0xeb, 0x6f, 0xee, 0xec, 0xff, 0xa5, 0xd8, 0x1f, 0xfd, 0xe1, 0x83, 0x56, 0x2f,
	0x7b, 0x8a, 0x76, 0x33, 0x90, 0xd3, 0x71, 0xc1, 0xdb, 0xf3, 0x0f, 0xda, 0x32, 0x2f, 0xeb, 0xd2,
	0x91, 0x48, 0x82, 0x35, 0xd6, 0x7b, 0x88, 0xb8, 0xd1, 0x62, 0x88, 0xd6, 0xc9, 0x78, 0x32, 0x39,
	0x19, 0x75, 0x0d, 0x66, 0x63, 0x67, 0x38, 0x7e, 0xd1, 0x85, 0xde, 0xc7, 0x1d, 0xec, 0xd6, 0x13,
	0x8c, 0xb2, 0x5c, 0x2d, 0x6b, 0x4b, 0x8f, 0xd1, 0x8c, 0xab, 0x15, 0x91, 0x9b, 0x7d, 0xff, 0x4e,
	0xfb, 0x9e, 0xa3, 0x30, 0x0f, 0x27, 0xcb, 0x85, 0x08, 0x6a, 0x84, 0xf9, 0x68, 0xd2, 0x13, 0x91,
	0xb5, 0x3d, 0xff, 0x7e, 0x9b, 0x9d, 0xd0, 0xf1, 0xac, 0x2a, 0x93, 0x70, 0x50, 0xa3, 0xec, 0x2e,
	0xde, 0xa0, 0x75, 0x9d, 0xa7, 0x31, 0x59, 0x31, 0x03, 0x9b, 0xe2, 0x71, 0xcc, 0x0e, 0xd0, 0x92,
	0x49, 0xa2, 0x45, 0xee, 0xec, 0xba, 0xe0, 0x75, 0x82, 0x26, 0x62, 0x0c, 0x77, 0x75, 0xfa, 0x5e,
	0x38, 0x26, 0x65, 0xe9, 0x5c, 0xb1, 0x91, 0x8a, 0x0e, 0xfd, 0xc8, 0xb1, 0x5c, 0xf0, 0xec, 0xa0,
	0x89, 0xd8, 0x33, 0xb4, 0xe8, 0x1f, 0x68, 0xc7, 0x76, 0x3b, 0x5b, 0x66, 0xaa, 0xea, 0xf5, 0x4c,
	0x0d, 0x3b, 0x7c, 0x75, 0xb9, 0xe2, 0xc6, 0xb7, 0x15, 0x37, 0xae, 0x56, 0x1c, 0x3e, 0x94, 0x1c,
	0x3e, 0x97, 0x1c, 0xbe, 0x94, 0x1c, 0x2e, 0x4b, 0x0e, 0xdf, 0x4b, 0x0e, 0x3f, 0x4b, 0x6e, 0x5c,
	0x95, 0x1c, 0x3e, 0xfd, 0xe0, 0x06, 0xde, 0x96, 0xea, 0x75, 0x5b, 0xb6, 0xc8, 0xd3, 0xd9, 0x90,
	0xd5, 0xe2, 0xf5, 0x3e, 0x49, 0x5d, 0x9f, 0xc2, 0x2f, 0x80, 0xa9, 0x45, 0x1f, 0xe8, 0xf0, 0x77,
	0x00, 0x00, 0x00, 0xff, 0xff, 0x0e, 0x3b, 0x54, 0x9f, 0x32, 0x03, 0x00, 0x00,
}
