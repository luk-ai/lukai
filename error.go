package lukai

// TrainingError returns the last training error that occured and clears it.
// I.e. calling TrainingError twice if there is only one error will return the
// error the first time and then nil afterwards.
// If there is no last error it returns nil.
func (mt *ModelType) TrainingError() error {
	mt.training.Lock()
	defer mt.training.Unlock()

	err := mt.training.err
	mt.training.err = nil
	return err
}

// ExamplesError returns the last examples saving error that occured and clears
// it. I.e. calling ExamplesError twice if there is only one error will return
// the error the first time and then nil afterwards.
// If there is no last error it returns nil.
func (mt *ModelType) ExamplesError() error {
	mt.examplesMeta.Lock()
	defer mt.examplesMeta.Unlock()

	err := mt.examplesMeta.err
	mt.examplesMeta.err = nil
	return err
}
