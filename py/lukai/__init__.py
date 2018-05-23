from __future__ import print_function
from __future__ import absolute_import

from .proto.aggregatorpb import aggregator_pb2
from .proto.clientpb import client_pb2
from .proto.managerpb import manager_pb2
from .proto.managerpb import manager_pb2_grpc
from . import saver

import tensorflow as tf
import grpc
import six
import srvlookup

import tempfile
import shutil
from os import path
import random

HyperParams = aggregator_pb2.HyperParams

# Metric reduces.
for k, v in client_pb2.MetricReduce.items():
    globals()[k] = v

# Event target types.
for k, v in client_pb2.Event.items():
    globals()[k] = v

def get_client(domain='manager.luk.ai'):
    records = srvlookup.lookup('grpclb', domain=domain)
    server = random.choice(records)
    addr = '{}:{}'.format(server.hostname, server.port)
    channel = grpc.secure_channel(addr, grpc.ssl_channel_credentials())
    return manager_pb2_grpc.ManagerStub(channel)

api_token = None
def set_api_token(token):
    global api_token
    api_token = token

def _v_to_array(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]

def _op_to_name(ops):
    return [op.name for op in ops]

def _get_ith(tpl, i):
    if i < len(tpl) and i >= 0:
        return tpl[i]
    return None


def upload(session, domain, model_type, hyper_params, metrics=None,
           event_targets=None, name="", description=""):
    metrics_proto = None
    if metrics is not None:
        metrics_proto = []
        for k, v in six.iteritems(metrics):
            metric_name = k.name
            if k.dtype != tf.float64:
                k = tf.cast(k, tf.float64)
            metrics_proto.append(client_pb2.Metric(fetch_name=k.name, reduce=v,
                                                   name=metric_name))

    events_proto = None
    if event_targets is not None:
        events_proto = {}
        for k, v in six.iteritems(event_targets):
            pre = _op_to_name(_v_to_array(_get_ith(v, 0)))
            post = _op_to_name(_v_to_array(_get_ith(v, 1)))
            events_proto[k] = client_pb2.EventTargets(pre=pre, post=post)

    dir = tempfile.mkdtemp("pok_model_upload_py")
    file_path = path.join(dir, "model.tar.gz")
    saver.save(session, target=file_path, metrics=metrics_proto,
               event_targets=events_proto)

    with open(file_path, 'rb') as content_file:
        content = content_file.read()
    shutil.rmtree(dir)

    print('Uploading model {}/{}/{}'.format(domain,model_type,name))
    client = get_client()
    resp = client.UploadModel(
        request=manager_pb2.UploadModelRequest(
            meta=manager_pb2.Model(
                domain=domain,
                model_type=model_type,
                name=name,
                description=description,
                hyper_params=hyper_params,
            ),
            model=content,
        ),
        metadata=[('token', api_token)],
    )
    print('Model uploaded! {}'.format(resp.model_url))
    return resp

