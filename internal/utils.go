package internal

import (
	"math"
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
)

const CurveSize = 34

type WeightingCurve [CurveSize]float32

var frequencyBands = WeightingCurve{
	0, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
	1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 9000, 10000, 12500, 15000, 20000, 25100,
}

// ComputeSpectrumWeights calculates the frequency weighting for a given frame size and sample rate
// based on the specified weighting curve.
func ComputeSpectrumWeights(frameSize int, sampleRate float64, curve WeightingCurve) []float64 {
	weights := make([]float64, frameSize/2+1)
	j := 1

	for i := range weights {
		frequency := float64(i) / float64(frameSize) * sampleRate
		for j < CurveSize-1 && frequency > float64(frequencyBands[j]) {
			j++
		}

		a0 := float64(curve[j-1])
		a1 := float64(curve[j])
		f0 := float64(frequencyBands[j-1])
		f1 := float64(frequencyBands[j])

		var weight float64
		switch f0 {
		case f1:
			weight = a0
		case 0:
			weight = (a1-a0)/f1*frequency + a0
		default:
			weight = (a1-a0)/(f1-f0)*frequency + (a0 - (a1-a0)/(f1/f0-1.0))
		}

		weights[i] = math.Pow(10, weight/20)
	}

	return weights
}

// CartesianToPolar converts a slice of complex numbers to polar coordinates,
// returning the magnitude and phase as separate slices.
func CartesianToPolar(complex []complex128) (magnitude []float64, phase []float64) {
	magnitude, phase = make([]float64, len(complex)), make([]float64, len(complex))

	for i, cnum := range complex {
		magnitude[i] = math.Sqrt(math.Pow(real(cnum), 2) + math.Pow(imag(cnum), 2))
		phase[i] = math.Atan2(imag(cnum), real(cnum))
	}

	return
}

// PrepareSpectrum applies a Hann window to the input frame and computes the FFT, making the result suitable for
// pitch detection with the YIN algorithm.
func PrepareSpectrum(frame []float64) []float64 {
	applyHannWindow(frame)

	complexSpectrum := fft.FFTReal(frame)

	spectrum := make([]float64, len(complexSpectrum)/2+1)
	for i := range spectrum {
		spectrum[i] = cmplx.Abs(complexSpectrum[i])
	}

	return spectrum
}

func applyHannWindow(frame []float64) {
	for i := range frame {
		frame[i] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(len(frame)-1)))
	}
}
