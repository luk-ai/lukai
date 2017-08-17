package debounce

import "time"

// Debounce calls a function `f` interval after call() is invoked.
func Debounce(interval time.Duration, f func()) (call func(), stop func()) {
	callChan := make(chan struct{}, 1)
	stopChan := make(chan struct{}, 1)
	go func() {
		var timeAfterChan <-chan time.Time
		for {
			select {
			case <-stopChan:
				return
			case <-callChan:
				if timeAfterChan == nil {
					timeAfterChan = time.After(interval)
				}
			case <-timeAfterChan:
				f()
				timeAfterChan = nil
			}
		}
	}()

	return func() {
			callChan <- struct{}{}
		}, func() {
			stopChan <- struct{}{}
		}
}
