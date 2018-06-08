package units

import "time"

type Bytes int64
type Bitsps int64

const (
	B  Bytes = 1
	KB       = 1000 * B
	MB       = 1000 * KB
	GB       = 1000 * MB
	TB       = 1000 * GB
	PB       = 1000 * TB

	BytesToBits = 8

	Bps  Bitsps = 1
	Kbps        = 1000 * Bps
	Mbps        = 1000 * Kbps
	Gbps        = 1000 * Mbps
	Tbps        = 1000 * Gbps
	Pbps        = 1000 * Tbps
)

const (
	Day = 24 * time.Hour
)
