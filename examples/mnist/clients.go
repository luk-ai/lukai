package main

import (
	"flag"
	"log"
	"os"
	"path"
	"strconv"
	"sync"

	"github.com/d4l3k/pok"
)

var (
	numClients = flag.Int("num_clients", 10, "the number of clients to simulate")
	modelType  = flag.String("model_type", "mnist", "the name of the model type")
	domain     = flag.String("domain", "", "the name of the domain")
)

const (
	perm    os.FileMode = 0755
	dataDir             = "tmp/data"
)

func main() {
	log.SetFlags(log.Flags() | log.Lshortfile)
	flag.Parse()

	if err := os.MkdirAll(dataDir, perm); err != nil {
		log.Fatal(err)
	}
	var clients []*pok.ModelType
	for i := 0; i < *numClients; i++ {
		mt, err := pok.MakeModelType(*domain, *modelType, path.Join(dataDir, strconv.Itoa(i)))
		if err != nil {
			log.Fatal(err)
		}
		clients = append(clients, mt)
	}
	var wg sync.WaitGroup

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
