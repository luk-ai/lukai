// bindable is a wrapper around lukai that is gomobile/gobind compatible.
//
// gobind is very restricted in what types can be used. Thus, this is a wrapper
// that uses byte arrays for types that can't be used. This isn't intended to be
// used directly, instead should be used via a language specific client wrapper.
//
// See github.com/luk-ai/lukai for details of the functions.
package bindable

import "github.com/luk-ai/lukai"

type ModelType struct {
	mt *lukai.ModelType
}

func MakeModelType(domain, modelType, dataDir string) (ModelType, error) {
	mt, err := lukai.MakeModelType(domain, modelType, dataDir)
	if err != nil {
		return ModelType{}, err
	}
	return ModelType{
		mt: mt,
	}, nil
}

func (mt ModelType) StartTraining() error {
	return mt.mt.StartTraining()
}

func (mt ModelType) StopTraining() error {
	return mt.mt.StopTraining()
}

func (mt ModelType) TotalExamples() int64 {
	return mt.mt.TotalExamples()
}

func (mt ModelType) Log(feeds, targets []byte) error {
	return nil
}

func (mt ModelType) Run(feeds, fetches, targets []byte) error {
	return nil
}
