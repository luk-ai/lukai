package tf

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
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
	SaverDefName           = "saver_def.pb"
	GraphDefName           = "graph_def.pb"
	SavedModelName         = "saved_model"
	TrainableVariablesName = "trainable_variables.json"
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
	dir, err := ioutil.TempDir("", "pok_model_load")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	gr, err := gzip.NewReader(reader)
	if err != nil {
		return nil, err
	}

	model := Model{
		Graph: tensorflow.NewGraph(),
	}

	var foundSaverDef, foundModel bool

	tr := tar.NewReader(gr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		if header.Name == GraphDefName {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(tr); err != nil {
				return nil, err
			}
			// Construct an in-memory graph from the serialized form.
			if err := model.Graph.Import(buf.Bytes(), ""); err != nil {
				return nil, err
			}
			foundModel = true
		} else if header.Name == SaverDefName {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(tr); err != nil {
				return nil, err
			}
			if err := proto.Unmarshal(buf.Bytes(), &model.SaverDef); err != nil {
				return nil, err
			}
			foundSaverDef = true
		} else if header.Name == TrainableVariablesName {
			if err := json.NewDecoder(tr).Decode(&model.TrainableVariables); err != nil {
				return nil, err
			}
		} else {
			file := path.Join(dir, header.Name)
			f, err := os.OpenFile(file, os.O_CREATE|os.O_WRONLY, 0755)
			if err != nil {
				return nil, err
			}
			_, err = io.Copy(f, tr)
			if err := f.Close(); err != nil {
				return nil, err
			}
			if err != nil {
				return nil, err
			}
		}
	}

	// Make sure we have all the information required.
	if !foundSaverDef {
		return nil, errors.New("missing " + SaverDefName)
	}
	if !foundModel {
		return nil, errors.New("missing " + GraphDefName)
	}

	// Create a session for inference over graph.
	model.Session, err = tensorflow.NewSession(model.Graph, nil)
	if err != nil {
		return nil, err
	}

	filenameTensor, err := tensorflow.NewTensor(path.Join(dir, SavedModelName))
	if err != nil {
		return nil, err
	}

	filenameOp, filenameOpI, err := ParseNodeOutput(model.SaverDef.FilenameTensorName)
	if err != nil {
		return nil, err
	}

	restoreOp, _, err := ParseNodeOutput(model.SaverDef.RestoreOpName)
	if err != nil {
		return nil, err
	}

	if _, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation(filenameOp).Output(filenameOpI): filenameTensor,
		},
		nil,
		[]*tensorflow.Operation{
			model.Graph.Operation(restoreOp),
		},
	); err != nil {
		return nil, err
	}

	return &model, nil
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
			Mode: 0755,
			Size: int64(buf.Len()),
		}); err != nil {
			return err
		}
		if _, err := buf.WriteTo(tw); err != nil {
			return err
		}
	}

	{
		buf, err := json.Marshal(model.TrainableVariables)
		if err != nil {
			return err
		}
		if err := tw.WriteHeader(&tar.Header{
			Name: TrainableVariablesName,
			Mode: 0755,
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
			model.Graph.Operation(filenameOp).Output(filenameOpI): filenameTensor,
		},
		nil,
		[]*tensorflow.Operation{
			model.Graph.Operation(saveOp),
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
		file, err := os.OpenFile(path, os.O_RDONLY, 0755)
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
