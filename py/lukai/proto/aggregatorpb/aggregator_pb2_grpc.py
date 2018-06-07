# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from lukai.proto.aggregatorpb import aggregator_pb2 as aggregatorpb_dot_aggregator__pb2


class AggregatorStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetWork = channel.unary_stream(
        '/aggregatorpb.Aggregator/GetWork',
        request_serializer=aggregatorpb_dot_aggregator__pb2.GetWorkRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.GetWorkResponse.FromString,
        )
    self.ReportWork = channel.stream_unary(
        '/aggregatorpb.Aggregator/ReportWork',
        request_serializer=aggregatorpb_dot_aggregator__pb2.ReportWorkRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.ReportWorkResponse.FromString,
        )
    self.Notify = channel.unary_unary(
        '/aggregatorpb.Aggregator/Notify',
        request_serializer=aggregatorpb_dot_aggregator__pb2.NotifyRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.NotifyResponse.FromString,
        )
    self.CancelModelTraining = channel.unary_unary(
        '/aggregatorpb.Aggregator/CancelModelTraining',
        request_serializer=aggregatorpb_dot_aggregator__pb2.CancelModelTrainingRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.CancelModelTrainingResponse.FromString,
        )


class AggregatorServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetWork(self, request, context):
    """GetWork sends work to clients to process.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ReportWork(self, request_iterator, context):
    """ReportWork is used to report the trained model/work to the server.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Notify(self, request, context):
    """Internal RPCs.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CancelModelTraining(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AggregatorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetWork': grpc.unary_stream_rpc_method_handler(
          servicer.GetWork,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.GetWorkRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.GetWorkResponse.SerializeToString,
      ),
      'ReportWork': grpc.stream_unary_rpc_method_handler(
          servicer.ReportWork,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.ReportWorkRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.ReportWorkResponse.SerializeToString,
      ),
      'Notify': grpc.unary_unary_rpc_method_handler(
          servicer.Notify,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.NotifyRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.NotifyResponse.SerializeToString,
      ),
      'CancelModelTraining': grpc.unary_unary_rpc_method_handler(
          servicer.CancelModelTraining,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.CancelModelTrainingRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.CancelModelTrainingResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'aggregatorpb.Aggregator', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class EdgeStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ProdModel = channel.unary_unary(
        '/aggregatorpb.Edge/ProdModel',
        request_serializer=aggregatorpb_dot_aggregator__pb2.ProdModelRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.ProdModelResponse.FromString,
        )
    self.FindWork = channel.unary_unary(
        '/aggregatorpb.Edge/FindWork',
        request_serializer=aggregatorpb_dot_aggregator__pb2.FindWorkRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.FindWorkResponse.FromString,
        )
    self.ModelURL = channel.unary_unary(
        '/aggregatorpb.Edge/ModelURL',
        request_serializer=aggregatorpb_dot_aggregator__pb2.ModelURLRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.ModelURLResponse.FromString,
        )
    self.ReportError = channel.unary_unary(
        '/aggregatorpb.Edge/ReportError',
        request_serializer=aggregatorpb_dot_aggregator__pb2.ReportErrorRequest.SerializeToString,
        response_deserializer=aggregatorpb_dot_aggregator__pb2.ReportErrorResponse.FromString,
        )


class EdgeServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def ProdModel(self, request, context):
    """ProdModel returns the current production model.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def FindWork(self, request, context):
    """FindWork returns an address of the aggregator that the client should
    request work from.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ModelURL(self, request, context):
    """ModelURL returns a URL that can be used to download the model. For billing
    purposes, hitting this endpoint will count as one download of the model.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ReportError(self, request, context):
    """ReportError reports an error to the server so the developers can later view
    them.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_EdgeServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ProdModel': grpc.unary_unary_rpc_method_handler(
          servicer.ProdModel,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.ProdModelRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.ProdModelResponse.SerializeToString,
      ),
      'FindWork': grpc.unary_unary_rpc_method_handler(
          servicer.FindWork,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.FindWorkRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.FindWorkResponse.SerializeToString,
      ),
      'ModelURL': grpc.unary_unary_rpc_method_handler(
          servicer.ModelURL,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.ModelURLRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.ModelURLResponse.SerializeToString,
      ),
      'ReportError': grpc.unary_unary_rpc_method_handler(
          servicer.ReportError,
          request_deserializer=aggregatorpb_dot_aggregator__pb2.ReportErrorRequest.FromString,
          response_serializer=aggregatorpb_dot_aggregator__pb2.ReportErrorResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'aggregatorpb.Edge', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
