package yinfft_test

import (
	"fmt"
	"iter"
	"log/slog"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/FreibergVlad/go-yinfft"
	"github.com/go-audio/wav"
)

func TestDetectFromFrame_WAV(t *testing.T) {
	t.Parallel()

	tests := []struct {
		filename      string
		wantFrequency float64
	}{
		{"testdata/Alesis-Fusion-Clean-Guitar-C3.wav", 130.81},
		{"testdata/Yamaha-TG500-GT-Nylon-E2.wav", 82.41},
	}

	frequencyThreshold := 1.0
	confidenceThreshold := 0.9

	type testResult struct {
		frequency  float64
		confidence float64
	}

	pitchDetector := pitchDetector(t)

	for _, test := range tests {
		t.Run(test.filename, func(t *testing.T) {
			t.Parallel()

			frames, err := framesFromWAV(test.filename, yinfft.DefaultParams.FrameSize)
			if err != nil {
				t.Fatalf("error reading .wav file %s: %v", test.filename, err)
			}

			testResults := []testResult{}
			for chunk := range frames {
				freq, conf, err := pitchDetector.DetectFromFrame(chunk)
				if err != nil {
					t.Fatalf("error detecting pitch for a frame: %v", err)
				}
				testResults = append(testResults, testResult{frequency: freq, confidence: conf})
			}

			testPassed := slices.ContainsFunc(testResults, func(result testResult) bool {
				return math.Abs(result.frequency-test.wantFrequency) < frequencyThreshold &&
					result.confidence >= confidenceThreshold
			})

			if !testPassed {
				t.Errorf(
					"incorrect frequency for %s, want %.2f Hz, got %v",
					test.filename, test.wantFrequency, testResults,
				)
			}
		})
	}
}
func TestDetectFromFrame_SineWaves(t *testing.T) {
	t.Parallel()

	frequencies := []float64{73.42, 82.41, 110, 146.83, 196, 246.94, 329.63}
	frequencyThreshold := 1.0
	confidenceThreshold := 0.9
	pitchDetector := pitchDetector(t)

	for _, wantFrequency := range frequencies {
		t.Run(fmt.Sprintf("running for sine wave %.2f Hz", wantFrequency), func(t *testing.T) {
			t.Parallel()

			frame := generateSineWave(wantFrequency, yinfft.DefaultParams.SampleRate, yinfft.DefaultParams.FrameSize)
			frequency, confidence, err := pitchDetector.DetectFromFrame(frame)
			if err != nil {
				t.Fatalf("error detecting pitch for a frame: %v", err)
			}

			if confidence < confidenceThreshold {
				t.Errorf("confidence is to low: got %.2f, want at least %.2f", confidence, confidenceThreshold)
			}

			if math.Abs(frequency-wantFrequency) >= frequencyThreshold {
				t.Errorf("incorrect frequency, got %.2f Hz, want %.2f Hz", frequency, wantFrequency)
			}
		})
	}
}

func generateSineWave(freq, sampleRate float64, length int) []float64 {
	signal := make([]float64, length)
	for i := range signal {
		signal[i] = math.Sin(2 * math.Pi * freq * float64(i) / sampleRate)
	}
	return signal
}

func framesFromWAV(filename string, chunkLen int) (iter.Seq[[]float64], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, decoder.Err()
	}

	buffer, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, err
	}

	return func(yield func([]float64) bool) {
		for chunk := range slices.Chunk(buffer.AsFloatBuffer().Data, chunkLen) {
			if len(chunk) < chunkLen {
				continue
			}
			if !yield(chunk) {
				return
			}
		}
	}, nil
}

func pitchDetector(t *testing.T) *yinfft.PitchDetector {
	t.Helper()

	params := yinfft.DefaultParams
	params.Logger = slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	pitchDetector, err := yinfft.New(params)
	if err != nil {
		t.Fatalf("error creating pitch detector: %v", err)
	}
	return pitchDetector
}
