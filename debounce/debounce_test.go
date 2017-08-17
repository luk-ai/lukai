package debounce

import (
	"sync"
	"testing"
	"time"
)

func TestDebounce(t *testing.T) {
	var called struct {
		sync.Mutex

		count int
	}
	call, stop := Debounce(1*time.Millisecond, func() {
		called.Lock()
		defer called.Unlock()
		called.count += 1
	})
	defer stop()

	for i := 0; i < 10; i++ {
		call()
	}
	time.Sleep(4 * time.Millisecond)
	called.Lock()
	if called.count != 1 {
		t.Fatalf("called.count = %d; not 1", called.count)
	}
	called.count = 0
	called.Unlock()

	start := time.Now()
	for time.Now().Sub(start) < 40*time.Millisecond {
		call()
	}
	time.Sleep(4 * time.Millisecond)
	called.Lock()
	if called.count <= 5 {
		called.count = 0
		t.Fatalf("called.count = %d; not > 5", called.count)
	}
	called.Unlock()

	call()
	time.Sleep(4 * time.Millisecond)
	called.Lock()
	if called.count == 1 {
		t.Fatalf("called.count = %d; not 1", called.count)
	}
	called.Unlock()
}
