from __future__ import print_function

from libpok.proto.aggregatorpb import aggregator_pb2
from libpok.proto.managerpb import manager_pb2
from libpok.proto.managerpb import manager_pb2_grpc
from libpok.proto.uipb import ui_pb2
from libpok import saver

import grpc

import tempfile
import shutil
from os import path

HyperParams = aggregator_pb2.HyperParams

def get_client():
    channel = grpc.insecure_channel('localhost:5002')
    return manager_pb2_grpc.ManagerStub(channel)

api_token = None
def set_api_token(token):
    global api_token
    api_token = token

def upload(sess, domain, model_type, hyper_params, name="", description=""):
    dir = tempfile.mkdtemp("pok_model_upload_py")
    file_path = path.join(dir, "model.tar.gz")
    saver.save(sess, target=file_path)
    with open(file_path, 'rb') as content_file:
        content = content_file.read()
    shutil.rmtree(dir)

    print('Uploading model {}/{}/{}'.format(domain,model_type,name))
    client = get_client()
    resp = client.UploadModel(
        request=manager_pb2.UploadModelRequest(
            meta=ui_pb2.Model(
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

