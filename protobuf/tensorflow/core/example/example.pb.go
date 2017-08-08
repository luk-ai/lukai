// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: protobuf/tensorflow/core/example/example.proto

/*
	Package tensorflow is a generated protocol buffer package.

	It is generated from these files:
		protobuf/tensorflow/core/example/example.proto

	It has these top-level messages:
		Example
		SequenceExample
*/
package tensorflow

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"
import tensorflow1 "github.com/d4l3k/pok/protobuf/tensorflow/core/example"

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

type Example struct {
	Features *tensorflow1.Features `protobuf:"bytes,1,opt,name=features" json:"features,omitempty"`
}

func (m *Example) Reset()                    { *m = Example{} }
func (*Example) ProtoMessage()               {}
func (*Example) Descriptor() ([]byte, []int) { return fileDescriptorExample, []int{0} }

func (m *Example) GetFeatures() *tensorflow1.Features {
	if m != nil {
		return m.Features
	}
	return nil
}

type SequenceExample struct {
	Context      *tensorflow1.Features     `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
	FeatureLists *tensorflow1.FeatureLists `protobuf:"bytes,2,opt,name=feature_lists,json=featureLists" json:"feature_lists,omitempty"`
}

func (m *SequenceExample) Reset()                    { *m = SequenceExample{} }
func (*SequenceExample) ProtoMessage()               {}
func (*SequenceExample) Descriptor() ([]byte, []int) { return fileDescriptorExample, []int{1} }

func (m *SequenceExample) GetContext() *tensorflow1.Features {
	if m != nil {
		return m.Context
	}
	return nil
}

func (m *SequenceExample) GetFeatureLists() *tensorflow1.FeatureLists {
	if m != nil {
		return m.FeatureLists
	}
	return nil
}

func init() {
	proto.RegisterType((*Example)(nil), "tensorflow.Example")
	proto.RegisterType((*SequenceExample)(nil), "tensorflow.SequenceExample")
}
func (this *Example) Equal(that interface{}) bool {
	if that == nil {
		if this == nil {
			return true
		}
		return false
	}

	that1, ok := that.(*Example)
	if !ok {
		that2, ok := that.(Example)
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
	if !this.Features.Equal(that1.Features) {
		return false
	}
	return true
}
func (this *SequenceExample) Equal(that interface{}) bool {
	if that == nil {
		if this == nil {
			return true
		}
		return false
	}

	that1, ok := that.(*SequenceExample)
	if !ok {
		that2, ok := that.(SequenceExample)
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
	if !this.Context.Equal(that1.Context) {
		return false
	}
	if !this.FeatureLists.Equal(that1.FeatureLists) {
		return false
	}
	return true
}
func (this *Example) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&tensorflow.Example{")
	if this.Features != nil {
		s = append(s, "Features: "+fmt.Sprintf("%#v", this.Features)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *SequenceExample) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&tensorflow.SequenceExample{")
	if this.Context != nil {
		s = append(s, "Context: "+fmt.Sprintf("%#v", this.Context)+",\n")
	}
	if this.FeatureLists != nil {
		s = append(s, "FeatureLists: "+fmt.Sprintf("%#v", this.FeatureLists)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringExample(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *Example) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Example) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.Features != nil {
		dAtA[i] = 0xa
		i++
		i = encodeVarintExample(dAtA, i, uint64(m.Features.Size()))
		n1, err := m.Features.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n1
	}
	return i, nil
}

func (m *SequenceExample) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SequenceExample) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.Context != nil {
		dAtA[i] = 0xa
		i++
		i = encodeVarintExample(dAtA, i, uint64(m.Context.Size()))
		n2, err := m.Context.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n2
	}
	if m.FeatureLists != nil {
		dAtA[i] = 0x12
		i++
		i = encodeVarintExample(dAtA, i, uint64(m.FeatureLists.Size()))
		n3, err := m.FeatureLists.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n3
	}
	return i, nil
}

func encodeFixed64Example(dAtA []byte, offset int, v uint64) int {
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
func encodeFixed32Example(dAtA []byte, offset int, v uint32) int {
	dAtA[offset] = uint8(v)
	dAtA[offset+1] = uint8(v >> 8)
	dAtA[offset+2] = uint8(v >> 16)
	dAtA[offset+3] = uint8(v >> 24)
	return offset + 4
}
func encodeVarintExample(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *Example) Size() (n int) {
	var l int
	_ = l
	if m.Features != nil {
		l = m.Features.Size()
		n += 1 + l + sovExample(uint64(l))
	}
	return n
}

func (m *SequenceExample) Size() (n int) {
	var l int
	_ = l
	if m.Context != nil {
		l = m.Context.Size()
		n += 1 + l + sovExample(uint64(l))
	}
	if m.FeatureLists != nil {
		l = m.FeatureLists.Size()
		n += 1 + l + sovExample(uint64(l))
	}
	return n
}

func sovExample(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozExample(x uint64) (n int) {
	return sovExample(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *Example) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&Example{`,
		`Features:` + strings.Replace(fmt.Sprintf("%v", this.Features), "Features", "tensorflow1.Features", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *SequenceExample) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&SequenceExample{`,
		`Context:` + strings.Replace(fmt.Sprintf("%v", this.Context), "Features", "tensorflow1.Features", 1) + `,`,
		`FeatureLists:` + strings.Replace(fmt.Sprintf("%v", this.FeatureLists), "FeatureLists", "tensorflow1.FeatureLists", 1) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringExample(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *Example) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowExample
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
			return fmt.Errorf("proto: Example: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Example: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Features", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowExample
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
				return ErrInvalidLengthExample
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Features == nil {
				m.Features = &tensorflow1.Features{}
			}
			if err := m.Features.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipExample(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthExample
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
func (m *SequenceExample) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowExample
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
			return fmt.Errorf("proto: SequenceExample: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SequenceExample: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Context", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowExample
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
				return ErrInvalidLengthExample
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Context == nil {
				m.Context = &tensorflow1.Features{}
			}
			if err := m.Context.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FeatureLists", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowExample
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
				return ErrInvalidLengthExample
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.FeatureLists == nil {
				m.FeatureLists = &tensorflow1.FeatureLists{}
			}
			if err := m.FeatureLists.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipExample(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthExample
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
func skipExample(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowExample
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
					return 0, ErrIntOverflowExample
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
					return 0, ErrIntOverflowExample
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
				return 0, ErrInvalidLengthExample
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowExample
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
				next, err := skipExample(dAtA[start:])
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
	ErrInvalidLengthExample = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowExample   = fmt.Errorf("proto: integer overflow")
)

func init() {
	proto.RegisterFile("protobuf/tensorflow/core/example/example.proto", fileDescriptorExample)
}

var fileDescriptorExample = []byte{
	// 243 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xd2, 0x2b, 0x28, 0xca, 0x2f,
	0xc9, 0x4f, 0x2a, 0x4d, 0xd3, 0x2f, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7,
	0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x4f, 0xad, 0x48, 0xcc, 0x2d, 0xc8, 0x81, 0xd3, 0x10, 0x85, 0x42,
	0x5c, 0x08, 0x65, 0x52, 0xaa, 0xb8, 0xb4, 0xa4, 0xa5, 0x26, 0x96, 0x94, 0x16, 0x41, 0xb5, 0x28,
	0x59, 0x73, 0xb1, 0xbb, 0x42, 0x24, 0x84, 0x0c, 0xb8, 0x38, 0xa0, 0x72, 0xc5, 0x12, 0x8c, 0x0a,
	0x8c, 0x1a, 0xdc, 0x46, 0x22, 0x7a, 0x08, 0x43, 0xf4, 0xdc, 0xa0, 0x72, 0x41, 0x70, 0x55, 0x4a,
	0x0d, 0x8c, 0x5c, 0xfc, 0xc1, 0xa9, 0x85, 0xa5, 0xa9, 0x79, 0xc9, 0xa9, 0x30, 0x53, 0xf4, 0xb8,
	0xd8, 0x93, 0xf3, 0xf3, 0x4a, 0x52, 0x2b, 0x4a, 0xf0, 0x1a, 0x02, 0x53, 0x24, 0x64, 0xcb, 0xc5,
	0x0b, 0x35, 0x2f, 0x3e, 0x27, 0xb3, 0xb8, 0xa4, 0x58, 0x82, 0x09, 0xac, 0x4b, 0x02, 0x8b, 0x2e,
	0x1f, 0x90, 0x7c, 0x10, 0x4f, 0x1a, 0x12, 0xcf, 0x29, 0xe2, 0xc2, 0x43, 0x39, 0x86, 0x1b, 0x0f,
	0xe5, 0x18, 0x3e, 0x3c, 0x94, 0x63, 0x6c, 0x78, 0x24, 0xc7, 0xb8, 0xe2, 0x91, 0x1c, 0xe3, 0x89,
	0x47, 0x72, 0x8c, 0x17, 0x1e, 0xc9, 0x31, 0x3e, 0x78, 0x24, 0xc7, 0xf8, 0xe2, 0x91, 0x1c, 0xc3,
	0x87, 0x47, 0x72, 0x8c, 0x13, 0x1e, 0xcb, 0x31, 0x70, 0x89, 0xe5, 0x17, 0xa5, 0x23, 0x1b, 0x0c,
	0x0d, 0x13, 0x27, 0x5e, 0xa8, 0xeb, 0x03, 0x40, 0x61, 0x52, 0x1c, 0xc0, 0xf8, 0x83, 0x91, 0x31,
	0x89, 0x0d, 0x1c, 0x40, 0xc6, 0x80, 0x00, 0x00, 0x00, 0xff, 0xff, 0x39, 0x5d, 0x54, 0x7d, 0x85,
	0x01, 0x00, 0x00,
}
