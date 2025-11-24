// Package yinfft implements the YinFFT pitch detection algorithm, which is a variant of the Yin algorithm that uses
// Fast Fourier Transform (FFT) for efficient pitch detection.
package yinfft

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/FreibergVlad/go-yinfft/internal"
	"github.com/FreibergVlad/go-yinfft/internal/peakdetector"
	"github.com/mjibson/go-dsp/fft"
)

type logger interface {
	Debug(msg string, args ...any)
}

type (
	// Params defines configuration options for the YinFFT pitch detector.
	Params struct {
		FrameSize         int     // Length of the input audio frame in samples.
		SampleRate        float64 // Audio sampling rate in Hz.
		ShouldInterpolate bool    // Whether to apply interpolation to the detected frequency.
		Tolerance         float64 // Peak detection tolerance.
		WeightingType     string  // Type of weighting curve to apply (e.g., "A", "B", "C", "D", "CUSTOM").
		MinFrequency      float64 // Minimum detectable frequency in Hz.
		MaxFrequency      float64 // Maximum detectable frequency in Hz.
		Logger            logger  // Optional logger for debug messages.
	}
	// PitchDetector is the main structure for detecting pitch using the YinFFT algorithm.
	PitchDetector struct {
		params           Params
		weights          []float64
		minPeriodSamples int
		maxPeriodSamples int
		peakDetector     *peakdetector.PeakDetector
	}
)

var (
	weightingCurves = map[string]internal.WeightingCurve{
		"EMPTY": {},
		"CUSTOM": {
			-75.8, -70.1, -60.8, -52.1, -44.2, -37.5, -31.3, -25.6, -20.9, -16.5, -12.6, -9.6, -7.0, -4.7, -3.0, -1.8,
			-0.8, -0.2, 0.0, 0.5, 1.6, 3.2, 5.4, 7.8, 8.1, 5.3, -2.4, -11.1, -12.8, -12.2, -7.4, -17.8, -17.8, -17.8,
		},
		"A": {
			-148.6, -50.4, -44.8, -39.5, -34.5, -30.3, -26.2, -22.4, -19.1, -16.2, -13.2, -10.8, -8.7, -6.6, -4.8,
			-3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.6, -0.1, -1.1, -1.8, -2.5, -4.3, -6.0, -9.3, -12.4,
		},
		"B": {
			-96.4, -24.2, -20.5, -17.1, -14.1, -11.6, -9.4, -7.3, -5.6, -4.2, -2.9, -2.0, -1.4, -0.9, -0.5, -0.3, -0.1,
			0.0, 0.0, 0.0, 0.0, -0.1, -0.2, -0.4, -0.7, -1.2, -1.9, -2.9, -3.6, -4.3, -6.1, -7.8, -11.2, -14.2,
		},
		"C": {
			-52.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -3.7, -4.4, -6.2, -7.9, -11.3, -14.3,
		},
		"D": {
			-46.6, -20.6, -18.7, -16.7, -14.7, -12.8, -10.9, -8.9, -7.2, -5.6, -3.9, -2.6, -1.6, -0.8, -0.4, -0.3, -0.5,
			-0.6, 0.0, 1.9, 5.0, 7.9, 10.3, 11.5, 11.1, 9.6, 7.6, 5.5, 4.4, 3.4, 1.4, -0.2, -2.7, -4.7,
		},
	}
	availableWeightingTypes = slices.Collect(maps.Keys(weightingCurves))
	DefaultParams           = Params{
		FrameSize:         8192,
		SampleRate:        44100,
		ShouldInterpolate: true,
		Tolerance:         1,
		WeightingType:     "CUSTOM",
		MinFrequency:      20,
		MaxFrequency:      22050,
	}
)

// New creates a new PitchDetector instance using the provided Params.
func New(params Params) (*PitchDetector, error) {
	maxPeriodSamples := int(math.Min(math.Ceil(params.SampleRate/params.MinFrequency), float64(params.FrameSize/2)))
	minPeriodSamples := int(math.Min(math.Floor(params.SampleRate/params.MaxFrequency), float64(params.FrameSize/2)))

	if maxPeriodSamples <= minPeriodSamples {
		minDetectable := params.SampleRate / float64(params.FrameSize/2)
		return nil, fmt.Errorf("maxFrequency <= minFrequency or out of range; min detectable = %.2f Hz", minDetectable)
	}

	curve, ok := weightingCurves[strings.ToUpper(params.WeightingType)]
	if !ok {
		return nil, fmt.Errorf(
			"invalid 'weightingType': %s; available weighting types: %+q",
			params.WeightingType,
			availableWeightingTypes,
		)
	}

	peakDetector, err := peakdetector.New(
		peakdetector.Params{
			Range:             float64(params.FrameSize)/2 + 1,
			MaxPeaks:          1,
			MaxPosition:       float64(maxPeriodSamples),
			MinPosition:       float64(minPeriodSamples),
			Threshold:         math.Inf(-1),
			OrderBy:           peakdetector.PeakOrderByAmplitude,
			ShouldInterpolate: params.ShouldInterpolate,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize peak detection algorithm: %w", err)
	}

	return &PitchDetector{
		params:           params,
		weights:          internal.ComputeSpectrumWeights(params.FrameSize, params.SampleRate, curve),
		minPeriodSamples: minPeriodSamples,
		maxPeriodSamples: maxPeriodSamples,
		peakDetector:     peakDetector,
	}, nil
}

// NewWithDefaultParams creates a PitchDetector with built-in default settings.
func NewWithDefaultParams() (*PitchDetector, error) {
	return New(DefaultParams)
}

// DetectFromFrame applies windowing and FFT to the input audio frame, then detects the fundamental frequency.
// The input frame must match the configured FrameSize. Returns the detected frequency, confidence, and any error encountered.
func (pd *PitchDetector) DetectFromFrame(frame []float64) (frequency float64, confidence float64, err error) {
	if len(frame) != pd.params.FrameSize {
		return 0, 0, fmt.Errorf("invalid frame size: expected %d, got %d", pd.params.FrameSize, len(frame))
	}
	return pd.DetectFromSpectrum(internal.PrepareSpectrum(frame))
}

// DetectFromSpectrum detects the fundamental frequency assuming the input is a magnitude spectrum. The spectrum should
// be obtained via FFT, windowed with a Hann window and should represent FrameSize/2+1 bins. Returns the detected frequency,
// confidence, and any error encountered.
func (pd *PitchDetector) DetectFromSpectrum(spectrum []float64) (frequency float64, confidence float64, err error) {
	yinLen := pd.params.FrameSize/2 + 1
	if len(spectrum) != yinLen {
		return 0, 0, fmt.Errorf("invalid spectrum size: expected %d, got %d", yinLen, len(spectrum))
	}

	sqrMag, sum := make([]float64, pd.params.FrameSize), 0.0
	sqrMag[0] = math.Pow(float64(spectrum[0]), 2) * pd.weights[0]
	for i := 1; i < len(spectrum); i++ {
		sqrMag[i] = math.Pow(float64(spectrum[i]), 2) * pd.weights[i]
		sqrMag[pd.params.FrameSize-i] = sqrMag[i]
		sum += sqrMag[i]
	}
	sum *= 2

	if sum == 0 {
		return 0, 0, nil
	}

	magnitude, phase := internal.CartesianToPolar(fft.FFTReal(sqrMag))

	yin := make([]float64, yinLen)
	yin[0] = 1
	tmp := 0.0
	for i := 1; i < len(yin); i++ {
		yin[i] = sum - magnitude[i]*math.Cos(phase[i])
		tmp += yin[i]
		yin[i] *= float64(i) / tmp
	}

	if pd.params.Tolerance < 1.0 && slices.Min(yin) >= pd.params.Tolerance {
		return 0, 0, nil
	}

	var tau, yinMin float64
	if pd.params.ShouldInterpolate {
		for i := range yin {
			yin[i] = -yin[i]
		}
		positions, amplitudes, err := pd.peakDetector.DetectPeaks(yin)
		if err != nil {
			return 0, 0, fmt.Errorf("peak detection error: %v", err)
		}
		if len(positions) > 0 && len(amplitudes) > 0 {
			tau = positions[0]
			yinMin = -amplitudes[0]
		} else {
			return 0, 0, fmt.Errorf("no peaks found by peak detection algorithm")
		}
	} else {
		yinMin = yin[pd.minPeriodSamples]
		for i := pd.minPeriodSamples; i <= pd.maxPeriodSamples; i++ {
			if yin[i] < yinMin {
				tau = float64(i)
				yinMin = yin[i]
			}
		}
	}

	if tau != 0 {
		return pd.params.SampleRate / tau, 1 - yinMin, nil
	}

	return 0, 0, nil
}
