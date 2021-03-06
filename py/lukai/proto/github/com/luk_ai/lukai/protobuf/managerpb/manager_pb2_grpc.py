# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from github.com.luk_ai.lukai.protobuf.managerpb import manager_pb2 as github_dot_com_dot_luk__ai_dot_lukai_dot_protobuf_dot_managerpb_dot_manager__pb2


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
        request_serializer=github_dot_com_dot_luk__ai_dot_lukai_dot_protobuf_dot_managerpb_dot_manager__pb2.UploadModelRequest.SerializeToString,
        response_deserializer=github_dot_com_dot_luk__ai_dot_lukai_dot_protobuf_dot_managerpb_dot_manager__pb2.UploadModelResponse.FromString,
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
          request_deserializer=github_dot_com_dot_luk__ai_dot_lukai_dot_protobuf_dot_managerpb_dot_manager__pb2.UploadModelRequest.FromString,
          response_serializer=github_dot_com_dot_luk__ai_dot_lukai_dot_protobuf_dot_managerpb_dot_manager__pb2.UploadModelResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'managerpb.Manager', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
