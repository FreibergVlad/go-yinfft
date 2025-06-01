# go-yinfft

[![Go Report Card](https://goreportcard.com/badge/github.com/FreibergVlad/go-yinfft)](https://goreportcard.com/report/github.com/FreibergVlad/go-yinfft)

This is a Go port of the [YinFFT algorithm](https://essentia.upf.edu/reference/std_PitchYinFFT.html), originally implemented in the [Essentia library](https://github.com/MTG/essentia) by the Music Technology Group at Universitat Pompeu Fabra, Barcelona. It uses an FFT-based approach derived from the original [Yin](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf) algorithm, optimized for performance in the frequency domain. It is suitable for real-time applications like guitar tuners, pitch analyzers, and audio feature extraction.

## Installation

```bash
go get github.com/FreibergVlad/go-yinfft
```

## Usage

### Quick Start

```go
package main

import (
	"fmt"
	"github.com/FreibergVlad/go-yinfft"
)

func main() {
	detector, _ := yinfft.NewWithDefaultParams()
	var frame []float64 = getAudioFrame() // You must provide 4096-sample audio frame

	freq, confidence, _ := detector.DetectFromFrame(frame)
	if confidence > 0.8 {
		fmt.Printf("Detected pitch: %.2f Hz\n", freq)
	}
}
```

### Customization

```go
params := yinfft.Params{
	FrameSize:         4096,
	SampleRate:        44100,
	ShouldInterpolate: true,
	Tolerance:         1,
	WeightingType:     "CUSTOM",
	MinFrequency:      20,
	MaxFrequency:      22050,
}
detector, _ := yinfft.New(params)
```

## License
This library is released under the MIT License.
Original algorithm by Essentia, ported to Go with respect and attribution.