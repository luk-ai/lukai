package main

import (
	"compress/gzip"
	"encoding/binary"
	"flag"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync"

	"github.com/d4l3k/pok"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	numClients = flag.Int("num_clients", 10, "the number of clients to simulate")
	modelType  = flag.String("model_type", "mnist", "the name of the model type")
	domain     = flag.String("domain", "", "the name of the domain")
)

var (
	datasetsURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	trainImages = "train-images-idx3-ubyte.gz"
	trainLabels = "train-labels-idx1-ubyte.gz"
	testImages  = "t10k-images-idx3-ubyte.gz"
	testLabels  = "t10k-labels-idx1-ubyte.gz"
)

const (
	perm    os.FileMode = 0755
	dataDir             = "tmp/data"
)

var endian = binary.BigEndian

func loadLabels(url string) ([][]float32, error) {
	const numLables = 10

	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	reader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var magic uint32
	if err := binary.Read(reader, endian, &magic); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, errors.Errorf(
			"Invalid magic number %d in MINST label file: %s",
			magic, url,
		)
	}
	var numItems uint32
	if err := binary.Read(reader, endian, &numItems); err != nil {
		return nil, err
	}
	labels := make([][]float32, 0, numItems)
	for i := uint32(0); i < numItems; i++ {
		var label uint8
		if err := binary.Read(reader, endian, &label); err != nil {
			return nil, err
		}
		labelProbs := make([]float32, numLables)
		labelProbs[label] = 1.0
		labels = append(labels, labelProbs)
	}

	return labels, nil
}

func loadImages(url string) ([][]float32, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	reader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var magic uint32
	if err := binary.Read(reader, endian, &magic); err != nil {
		return nil, err
	}
	if magic != 2051 {
		return nil, errors.Errorf(
			"Invalid magic number %d in MINST image file: %s",
			magic, url,
		)
	}
	var numItems uint32
	if err := binary.Read(reader, endian, &numItems); err != nil {
		return nil, err
	}
	var rows uint32
	if err := binary.Read(reader, endian, &rows); err != nil {
		return nil, err
	}
	var cols uint32
	if err := binary.Read(reader, endian, &cols); err != nil {
		return nil, err
	}
	images := make([][]float32, 0, numItems)
	for i := uint32(0); i < numItems; i++ {
		imageBytes := make([]uint8, rows*cols)
		if err := binary.Read(reader, endian, &imageBytes); err != nil {
			return nil, err
		}
		image := make([]float32, rows*cols)
		for i, val := range imageBytes {
			image[i] = float32(val)
		}

		images = append(images, image)
	}

	return images, nil
}

func main() {
	log.SetFlags(log.Flags() | log.Lshortfile)
	flag.Parse()

	if err := os.MkdirAll(dataDir, perm); err != nil {
		log.Fatal(err)
	}
	needData := false
	var clients []*pok.ModelType
	for i := 0; i < *numClients; i++ {
		mt, err := pok.MakeModelType(*domain, *modelType, path.Join(dataDir, strconv.Itoa(i)))
		if err != nil {
			log.Fatal(err)
		}
		if mt.TotalExamples() == 0 {
			needData = true
		}
		clients = append(clients, mt)
	}
	var wg sync.WaitGroup

	if needData {
		dataImagesURL := datasetsURL + trainImages
		dataLabelsURL := datasetsURL + trainLabels
		log.Printf("Loading training data: %s, %s", dataImagesURL, dataLabelsURL)
		images, err := loadImages(dataImagesURL)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Loaded images! %d", len(images))
		labels, err := loadLabels(dataLabelsURL)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Loaded labels! %d", len(labels))
		if len(images) != len(labels) {
			log.Fatal("# images != # labels")
		}

		dropout, err := tensorflow.NewTensor(float32(0.5))
		if err != nil {
			log.Fatal(err)
		}

		for i, image := range images {
			client := clients[i%len(clients)]
			x, err := tensorflow.NewTensor([][]float32{image})
			if err != nil {
				log.Fatal(err)
			}
			y, err := tensorflow.NewTensor([][]float32{labels[i]})
			if err != nil {
				log.Fatal(err)
			}
			if err := client.Log(
				map[string]*tensorflow.Tensor{
					"Placeholder:0":         x,
					"Placeholder_1:0":       y,
					"dropout/Placeholder:0": dropout,
				},
				[]string{"adam_optimizer/Adam"},
			); err != nil {
				log.Fatal(err)
			}
		}
	}

	for i, mt := range clients {
		log.Printf("%d. StartTraining...", i)
		if err := mt.StartTraining(); err != nil {
			log.Fatal(err)
		}
		wg.Add(1)
		defer func() {
			if err := mt.StopTraining(); err != nil {
				log.Println(err)
			}
			wg.Done()
		}()
	}

	wg.Wait()
}
