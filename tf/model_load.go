package tf

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

const (
	SaverDefName   = "saver_def.pb"
	GraphDefName   = "graph_def.pb"
	SavedModelName = "saved_model"
	ModelMetaName  = "model_meta.pb"
	// FilePerm is the file permission all the model files use.
	FilePerm = 0600
)

// LoadModel loads a model from a provided .tar.gz io stream. The returned
// session must be closed when done using.
//
// The .tar.gz file should contain the following files:
// - saver_def.pb
// - graph_def.pb
// - checkpoint
// - saved_model-<iteration>.{index,meta,data-*}
func LoadModel(reader io.Reader) (*Model, error) {
	model := Model{}
	if err := model.Load(reader); err != nil {
		return nil, err
	}
	return &model, nil
}

func (model *Model) Load(reader io.Reader) error {
	dir, err := ioutil.TempDir("", "pok_model_load")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	gr, err := gzip.NewReader(reader)
	if err != nil {
		return err
	}
	defer gr.Close()

	var foundSaverDef, foundModel bool

	tr := tar.NewReader(gr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		if header.Name == GraphDefName {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(tr); err != nil {
				return err
			}
			if model.Graph == nil {
				model.Graph = tensorflow.NewGraph()
			}
			// Construct an in-memory graph from the serialized form.
			if err := model.Graph.Import(buf.Bytes(), model.Prefix); err != nil {
				return err
			}
			foundModel = true
		} else if header.Name == SaverDefName {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(tr); err != nil {
				return err
			}
			if err := proto.Unmarshal(buf.Bytes(), &model.SaverDef); err != nil {
				return err
			}
			foundSaverDef = true
		} else if header.Name == ModelMetaName {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(tr); err != nil {
				return err
			}
			if err := proto.Unmarshal(buf.Bytes(), &model.Meta); err != nil {
				return err
			}
		} else {
			file := path.Join(dir, header.Name)
			f, err := os.OpenFile(file, os.O_CREATE|os.O_WRONLY, FilePerm)
			if err != nil {
				return err
			}
			_, err = io.Copy(f, tr)
			if err := f.Close(); err != nil {
				return err
			}
			if err != nil {
				return err
			}
		}
	}

	// Make sure we have all the information required.
	if !foundSaverDef {
		return errors.New("missing " + SaverDefName)
	}
	if !foundModel {
		return errors.New("missing " + GraphDefName)
	}

	// Create a session for inference over graph.
	model.Session, err = tensorflow.NewSession(model.Graph, nil)
	if err != nil {
		return err
	}

	filenameTensor, err := tensorflow.NewTensor(path.Join(dir, SavedModelName))
	if err != nil {
		return err
	}

	filenameOp, filenameOpI, err := ParseNodeOutput(model.SaverDef.FilenameTensorName)
	if err != nil {
		return err
	}

	restoreOp, _, err := ParseNodeOutput(model.SaverDef.RestoreOpName)
	if err != nil {
		return err
	}

	if _, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation(model.ApplyPrefix(filenameOp)).Output(filenameOpI): filenameTensor,
		},
		nil,
		[]*tensorflow.Operation{
			model.Graph.Operation(model.ApplyPrefix(restoreOp)),
		},
	); err != nil {
		return err
	}

	return nil
}

// Save saves the model to the writer. See LoadModel.
func (model *Model) Save(writer io.Writer) error {
	gw := gzip.NewWriter(writer)
	defer gw.Close()

	tw := tar.NewWriter(gw)
	defer tw.Close()

	{
		buf, err := proto.Marshal(&model.SaverDef)
		if err != nil {
			return err
		}
		if err := tw.WriteHeader(&tar.Header{
			Name: SaverDefName,
			Size: int64(len(buf)),
		}); err != nil {
			return err
		}
		if _, err := tw.Write(buf); err != nil {
			return err
		}
	}

	{
		var buf bytes.Buffer
		if _, err := model.Graph.WriteTo(&buf); err != nil {
			return err
		}
		if err := tw.WriteHeader(&tar.Header{
			Name: GraphDefName,
			Mode: FilePerm,
			Size: int64(buf.Len()),
		}); err != nil {
			return err
		}
		if _, err := buf.WriteTo(tw); err != nil {
			return err
		}
	}

	{
		buf, err := proto.Marshal(&model.Meta)
		if err != nil {
			return err
		}
		if err := tw.WriteHeader(&tar.Header{
			Name: ModelMetaName,
			Mode: FilePerm,
			Size: int64(len(buf)),
		}); err != nil {
			return err
		}
		if _, err := tw.Write(buf); err != nil {
			return err
		}
	}

	dir, err := ioutil.TempDir("", "pok_model_save")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	// Create the filename tensor.
	filenameTensor, err := tensorflow.NewTensor(path.Join(dir, SavedModelName))
	if err != nil {
		return err
	}

	filenameOp, filenameOpI, err := ParseNodeOutput(model.SaverDef.FilenameTensorName)
	if err != nil {
		return err
	}

	saveOp, _, err := ParseNodeOutput(model.SaverDef.SaveTensorName)
	if err != nil {
		return err
	}

	// Run the saver.
	if _, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation(model.ApplyPrefix(filenameOp)).Output(filenameOpI): filenameTensor,
		},
		nil,
		[]*tensorflow.Operation{
			model.Graph.Operation(model.ApplyPrefix(saveOp)),
		},
	); err != nil {
		return err
	}

	// Walk all outputted files from the Saver and store them into the tar.
	if err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}

		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		if err := tw.WriteHeader(header); err != nil {
			return err
		}
		file, err := os.OpenFile(path, os.O_RDONLY, FilePerm)
		if err != nil {
			return errors.Wrapf(err, "failed to open file %q/%q", dir, path)
		}
		defer file.Close()
		if _, err := io.Copy(tw, file); err != nil {
			return errors.Wrapf(err, "failed to copy file %q/%q", dir, path)
		}
		return nil
	}); err != nil {
		return err
	}

	return nil
}
