import tensorflow as tf
import tempfile
import shutil
import tarfile
import os
from os import path

# Constants used for files inside the model archive.
SaverDefName   = "saver_def.pb"
GraphDefName   = "graph_def.pb"
SavedModelName = "saved_model"

# save saves the session to the specified target file path. This creates a
# format that Pok can read and load.
def save(sess, target="model.tar.gz"):
    dir = tempfile.mkdtemp("pok_model_save_py")

    saver = tf.train.Saver()

    tf.train.write_graph(sess.graph_def, dir, GraphDefName, as_text=False)
    saver.save(sess, path.join(dir, SavedModelName), global_step=0)
    saver_def = saver.as_saver_def().SerializeToString()

    with open(path.join(dir, SaverDefName), "wb") as file:
        file.write(saver_def)

    tar = tarfile.open(target, "w:gz")
    for file in os.listdir(dir):
        def keep_file_name(tarinfo):
            tarinfo.name = file
            return tarinfo
        tar.add(path.join(dir, file), filter=keep_file_name)
    tar.close()

    shutil.rmtree(dir)
