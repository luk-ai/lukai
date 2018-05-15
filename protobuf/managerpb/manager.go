package managerpb

func (m *UploadModelRequest) GetDomain() string {
	meta := m.GetMeta()
	return meta.GetDomain()
}
