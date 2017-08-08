// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: protobuf/tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.proto

/*
	Package tensorflow_decision_trees is a generated protocol buffer package.

	It is generated from these files:
		protobuf/tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.proto

	It has these top-level messages:
		MatchingValuesTest
*/
package tensorflow_decision_trees

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"
import tensorflow_decision_trees1 "github.com/d4l3k/pok/protobuf/tensorflow/contrib/decision_trees/proto"

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

// Used in generic_tree_model.BinaryNode.left_child_test.
// Tests whether the feature's value belongs to the specified list,
// (or does not belong if inverse=True).
type MatchingValuesTest struct {
	// When the feature is missing, the test's outcome is undefined.
	FeatureId *tensorflow_decision_trees1.FeatureId `protobuf:"bytes,1,opt,name=feature_id,json=featureId" json:"feature_id,omitempty"`
	Value     []*tensorflow_decision_trees1.Value   `protobuf:"bytes,2,rep,name=value" json:"value,omitempty"`
	Inverse   bool                                  `protobuf:"varint,3,opt,name=inverse,proto3" json:"inverse,omitempty"`
}

func (m *MatchingValuesTest) Reset()      { *m = MatchingValuesTest{} }
func (*MatchingValuesTest) ProtoMessage() {}
func (*MatchingValuesTest) Descriptor() ([]byte, []int) {
	return fileDescriptorGenericTreeModelExtensions, []int{0}
}

func (m *MatchingValuesTest) GetFeatureId() *tensorflow_decision_trees1.FeatureId {
	if m != nil {
		return m.FeatureId
	}
	return nil
}

func (m *MatchingValuesTest) GetValue() []*tensorflow_decision_trees1.Value {
	if m != nil {
		return m.Value
	}
	return nil
}

func (m *MatchingValuesTest) GetInverse() bool {
	if m != nil {
		return m.Inverse
	}
	return false
}

func init() {
	proto.RegisterType((*MatchingValuesTest)(nil), "tensorflow.decision_trees.MatchingValuesTest")
}
func (this *MatchingValuesTest) Equal(that interface{}) bool {
	if that == nil {
		if this == nil {
			return true
		}
		return false
	}

	that1, ok := that.(*MatchingValuesTest)
	if !ok {
		that2, ok := that.(MatchingValuesTest)
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
	if !this.FeatureId.Equal(that1.FeatureId) {
		return false
	}
	if len(this.Value) != len(that1.Value) {
		return false
	}
	for i := range this.Value {
		if !this.Value[i].Equal(that1.Value[i]) {
			return false
		}
	}
	if this.Inverse != that1.Inverse {
		return false
	}
	return true
}
func (this *MatchingValuesTest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&tensorflow_decision_trees.MatchingValuesTest{")
	if this.FeatureId != nil {
		s = append(s, "FeatureId: "+fmt.Sprintf("%#v", this.FeatureId)+",\n")
	}
	if this.Value != nil {
		s = append(s, "Value: "+fmt.Sprintf("%#v", this.Value)+",\n")
	}
	s = append(s, "Inverse: "+fmt.Sprintf("%#v", this.Inverse)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringGenericTreeModelExtensions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *MatchingValuesTest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *MatchingValuesTest) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.FeatureId != nil {
		dAtA[i] = 0xa
		i++
		i = encodeVarintGenericTreeModelExtensions(dAtA, i, uint64(m.FeatureId.Size()))
		n1, err := m.FeatureId.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n1
	}
	if len(m.Value) > 0 {
		for _, msg := range m.Value {
			dAtA[i] = 0x12
			i++
			i = encodeVarintGenericTreeModelExtensions(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if m.Inverse {
		dAtA[i] = 0x18
		i++
		if m.Inverse {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i++
	}
	return i, nil
}

func encodeFixed64GenericTreeModelExtensions(dAtA []byte, offset int, v uint64) int {
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
func encodeFixed32GenericTreeModelExtensions(dAtA []byte, offset int, v uint32) int {
	dAtA[offset] = uint8(v)
	dAtA[offset+1] = uint8(v >> 8)
	dAtA[offset+2] = uint8(v >> 16)
	dAtA[offset+3] = uint8(v >> 24)
	return offset + 4
}
func encodeVarintGenericTreeModelExtensions(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *MatchingValuesTest) Size() (n int) {
	var l int
	_ = l
	if m.FeatureId != nil {
		l = m.FeatureId.Size()
		n += 1 + l + sovGenericTreeModelExtensions(uint64(l))
	}
	if len(m.Value) > 0 {
		for _, e := range m.Value {
			l = e.Size()
			n += 1 + l + sovGenericTreeModelExtensions(uint64(l))
		}
	}
	if m.Inverse {
		n += 2
	}
	return n
}

func sovGenericTreeModelExtensions(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozGenericTreeModelExtensions(x uint64) (n int) {
	return sovGenericTreeModelExtensions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *MatchingValuesTest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&MatchingValuesTest{`,
		`FeatureId:` + strings.Replace(fmt.Sprintf("%v", this.FeatureId), "FeatureId", "tensorflow_decision_trees1.FeatureId", 1) + `,`,
		`Value:` + strings.Replace(fmt.Sprintf("%v", this.Value), "Value", "tensorflow_decision_trees1.Value", 1) + `,`,
		`Inverse:` + fmt.Sprintf("%v", this.Inverse) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringGenericTreeModelExtensions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *MatchingValuesTest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowGenericTreeModelExtensions
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
			return fmt.Errorf("proto: MatchingValuesTest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: MatchingValuesTest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FeatureId", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGenericTreeModelExtensions
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
				return ErrInvalidLengthGenericTreeModelExtensions
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.FeatureId == nil {
				m.FeatureId = &tensorflow_decision_trees1.FeatureId{}
			}
			if err := m.FeatureId.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Value", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGenericTreeModelExtensions
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
				return ErrInvalidLengthGenericTreeModelExtensions
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Value = append(m.Value, &tensorflow_decision_trees1.Value{})
			if err := m.Value[len(m.Value)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Inverse", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGenericTreeModelExtensions
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.Inverse = bool(v != 0)
		default:
			iNdEx = preIndex
			skippy, err := skipGenericTreeModelExtensions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthGenericTreeModelExtensions
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
func skipGenericTreeModelExtensions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowGenericTreeModelExtensions
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
					return 0, ErrIntOverflowGenericTreeModelExtensions
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
					return 0, ErrIntOverflowGenericTreeModelExtensions
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
				return 0, ErrInvalidLengthGenericTreeModelExtensions
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowGenericTreeModelExtensions
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
				next, err := skipGenericTreeModelExtensions(dAtA[start:])
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
	ErrInvalidLengthGenericTreeModelExtensions = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowGenericTreeModelExtensions   = fmt.Errorf("proto: integer overflow")
)

func init() {
	proto.RegisterFile("protobuf/tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.proto", fileDescriptorGenericTreeModelExtensions)
}

var fileDescriptorGenericTreeModelExtensions = []byte{
	// 268 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x0a, 0x29, 0x28, 0xca, 0x2f,
	0xc9, 0x4f, 0x2a, 0x4d, 0xd3, 0x2f, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7,
	0x4f, 0xce, 0xcf, 0x2b, 0x29, 0xca, 0x4c, 0xd2, 0x4f, 0x49, 0x4d, 0xce, 0x2c, 0xce, 0xcc, 0xcf,
	0x8b, 0x2f, 0x29, 0x4a, 0x4d, 0x2d, 0xd6, 0x07, 0x2b, 0xd5, 0x4f, 0x4f, 0xcd, 0x4b, 0x2d, 0xca,
	0x4c, 0x06, 0x8b, 0xc5, 0xe7, 0xe6, 0xa7, 0xa4, 0xe6, 0xc4, 0xa7, 0x56, 0x80, 0x34, 0x67, 0xe6,
	0xe7, 0x15, 0xeb, 0x81, 0xd5, 0x08, 0x49, 0x22, 0x0c, 0xd3, 0x43, 0x35, 0x44, 0xca, 0x81, 0x7c,
	0x7b, 0x20, 0x86, 0x2b, 0xad, 0x66, 0xe4, 0x12, 0xf2, 0x4d, 0x2c, 0x49, 0xce, 0xc8, 0xcc, 0x4b,
	0x0f, 0x4b, 0xcc, 0x29, 0x4d, 0x2d, 0x0e, 0x49, 0x2d, 0x2e, 0x11, 0x72, 0xe6, 0xe2, 0x4a, 0x4b,
	0x4d, 0x2c, 0x29, 0x2d, 0x4a, 0x8d, 0xcf, 0x4c, 0x91, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x36, 0x52,
	0xd1, 0xc3, 0xe9, 0x10, 0x3d, 0x37, 0x88, 0x62, 0xcf, 0x94, 0x20, 0xce, 0x34, 0x18, 0x53, 0xc8,
	0x8c, 0x8b, 0xb5, 0x0c, 0x64, 0xa4, 0x04, 0x93, 0x02, 0xb3, 0x06, 0xb7, 0x91, 0x02, 0x1e, 0xfd,
	0x60, 0xab, 0x83, 0x20, 0xca, 0x85, 0x24, 0xb8, 0xd8, 0x33, 0xf3, 0xca, 0x52, 0x8b, 0x8a, 0x53,
	0x25, 0x98, 0x15, 0x18, 0x35, 0x38, 0x82, 0x60, 0x5c, 0x27, 0x9d, 0x0b, 0x0f, 0xe5, 0x18, 0x6e,
	0x3c, 0x94, 0x63, 0xf8, 0xf0, 0x50, 0x8e, 0xb1, 0xe1, 0x91, 0x1c, 0xe3, 0x8a, 0x47, 0x72, 0x8c,
	0x27, 0x1e, 0xc9, 0x31, 0x5e, 0x78, 0x24, 0xc7, 0xf8, 0xe0, 0x91, 0x1c, 0xe3, 0x8b, 0x47, 0x72,
	0x0c, 0x1f, 0x1e, 0xc9, 0x31, 0x4e, 0x78, 0x2c, 0xc7, 0x90, 0xc4, 0x06, 0xf6, 0xa2, 0x31, 0x20,
	0x00, 0x00, 0xff, 0xff, 0x4c, 0xea, 0xd5, 0xdb, 0x97, 0x01, 0x00, 0x00,
}
