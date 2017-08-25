# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: managerpb/manager.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from github.com.gogo.protobuf.gogoproto import gogo_pb2 as github_dot_com_dot_gogo_dot_protobuf_dot_gogoproto_dot_gogo__pb2
from github.com.d4l3k.pok.protobuf.aggregatorpb import aggregator_pb2 as github_dot_com_dot_d4l3k_dot_pok_dot_protobuf_dot_aggregatorpb_dot_aggregator__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='managerpb/manager.proto',
  package='managerpb',
  syntax='proto3',
  serialized_pb=_b('\n\x17managerpb/manager.proto\x12\tmanagerpb\x1a-github.com/gogo/protobuf/gogoproto/gogo.proto\x1a;github.com/d4l3k/pok/protobuf/aggregatorpb/aggregator.proto\"\x85\x01\n\x05Model\x12\x0e\n\x06\x64omain\x18\x02 \x01(\t\x12\x12\n\nmodel_type\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x05 \x01(\t\x12\x35\n\x0chyper_params\x18\x08 \x01(\x0b\x32\x19.aggregatorpb.HyperParamsB\x04\xc8\xde\x1f\x00\"I\n\x12UploadModelRequest\x12$\n\x04meta\x18\x01 \x01(\x0b\x32\x10.managerpb.ModelB\x04\xc8\xde\x1f\x00\x12\r\n\x05model\x18\x02 \x01(\x0c\":\n\x13UploadModelResponse\x12\x10\n\x08model_id\x18\x01 \x01(\x04\x12\x11\n\tmodel_url\x18\x02 \x01(\t2Y\n\x07Manager\x12N\n\x0bUploadModel\x12\x1d.managerpb.UploadModelRequest\x1a\x1e.managerpb.UploadModelResponse\"\x00\x62\x06proto3')
  ,
  dependencies=[github_dot_com_dot_gogo_dot_protobuf_dot_gogoproto_dot_gogo__pb2.DESCRIPTOR,github_dot_com_dot_d4l3k_dot_pok_dot_protobuf_dot_aggregatorpb_dot_aggregator__pb2.DESCRIPTOR,])




_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='managerpb.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='domain', full_name='managerpb.Model.domain', index=0,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model_type', full_name='managerpb.Model.model_type', index=1,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='managerpb.Model.name', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='description', full_name='managerpb.Model.description', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hyper_params', full_name='managerpb.Model.hyper_params', index=4,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\310\336\037\000'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=147,
  serialized_end=280,
)


_UPLOADMODELREQUEST = _descriptor.Descriptor(
  name='UploadModelRequest',
  full_name='managerpb.UploadModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='meta', full_name='managerpb.UploadModelRequest.meta', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\310\336\037\000'))),
    _descriptor.FieldDescriptor(
      name='model', full_name='managerpb.UploadModelRequest.model', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=282,
  serialized_end=355,
)


_UPLOADMODELRESPONSE = _descriptor.Descriptor(
  name='UploadModelResponse',
  full_name='managerpb.UploadModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_id', full_name='managerpb.UploadModelResponse.model_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model_url', full_name='managerpb.UploadModelResponse.model_url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=357,
  serialized_end=415,
)

_MODEL.fields_by_name['hyper_params'].message_type = github_dot_com_dot_d4l3k_dot_pok_dot_protobuf_dot_aggregatorpb_dot_aggregator__pb2._HYPERPARAMS
_UPLOADMODELREQUEST.fields_by_name['meta'].message_type = _MODEL
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['UploadModelRequest'] = _UPLOADMODELREQUEST
DESCRIPTOR.message_types_by_name['UploadModelResponse'] = _UPLOADMODELRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
  DESCRIPTOR = _MODEL,
  __module__ = 'managerpb.manager_pb2'
  # @@protoc_insertion_point(class_scope:managerpb.Model)
  ))
_sym_db.RegisterMessage(Model)

UploadModelRequest = _reflection.GeneratedProtocolMessageType('UploadModelRequest', (_message.Message,), dict(
  DESCRIPTOR = _UPLOADMODELREQUEST,
  __module__ = 'managerpb.manager_pb2'
  # @@protoc_insertion_point(class_scope:managerpb.UploadModelRequest)
  ))
_sym_db.RegisterMessage(UploadModelRequest)

UploadModelResponse = _reflection.GeneratedProtocolMessageType('UploadModelResponse', (_message.Message,), dict(
  DESCRIPTOR = _UPLOADMODELRESPONSE,
  __module__ = 'managerpb.manager_pb2'
  # @@protoc_insertion_point(class_scope:managerpb.UploadModelResponse)
  ))
_sym_db.RegisterMessage(UploadModelResponse)


_MODEL.fields_by_name['hyper_params'].has_options = True
_MODEL.fields_by_name['hyper_params']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\310\336\037\000'))
_UPLOADMODELREQUEST.fields_by_name['meta'].has_options = True
_UPLOADMODELREQUEST.fields_by_name['meta']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\310\336\037\000'))
try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities


  class ManagerStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.UploadModel = channel.unary_unary(
          '/managerpb.Manager/UploadModel',
          request_serializer=UploadModelRequest.SerializeToString,
          response_deserializer=UploadModelResponse.FromString,
          )


  class ManagerServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def UploadModel(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_ManagerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'UploadModel': grpc.unary_unary_rpc_method_handler(
            servicer.UploadModel,
            request_deserializer=UploadModelRequest.FromString,
            response_serializer=UploadModelResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'managerpb.Manager', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaManagerServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def UploadModel(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaManagerStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def UploadModel(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    UploadModel.future = None


  def beta_create_Manager_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('managerpb.Manager', 'UploadModel'): UploadModelRequest.FromString,
    }
    response_serializers = {
      ('managerpb.Manager', 'UploadModel'): UploadModelResponse.SerializeToString,
    }
    method_implementations = {
      ('managerpb.Manager', 'UploadModel'): face_utilities.unary_unary_inline(servicer.UploadModel),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_Manager_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('managerpb.Manager', 'UploadModel'): UploadModelRequest.SerializeToString,
    }
    response_deserializers = {
      ('managerpb.Manager', 'UploadModel'): UploadModelResponse.FromString,
    }
    cardinalities = {
      'UploadModel': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'managerpb.Manager', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
