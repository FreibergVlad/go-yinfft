package peakdetector

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

type peak struct {
	position  float64
	magnitude float64
}

type PeakOrderBy string

const (
	PeakOrderByPosition  PeakOrderBy = "position"
	PeakOrderByAmplitude PeakOrderBy = "amplitude"
)

type Params struct {
	Range             float64
	MaxPeaks          int
	MaxPosition       float64
	MinPosition       float64
	Threshold         float64
	OrderBy           PeakOrderBy
	ShouldInterpolate bool
	MinPeakDistance   float64
}

type PeakDetector struct {
	params Params
}

func New(params Params) (*PeakDetector, error) {
	if params.MinPosition >= params.MaxPosition {
		return nil, fmt.Errorf("MinPosition must be less than MaxPosition")
	}
	if params.OrderBy != PeakOrderByPosition && params.OrderBy != PeakOrderByAmplitude {
		return nil, fmt.Errorf("invalid OrderBy value: %s, must be one of [%s, %s]", params.OrderBy, PeakOrderByPosition, PeakOrderByAmplitude)
	}
	return &PeakDetector{params: params}, nil
}

func (pd *PeakDetector) DetectPeaks(input []float64) (positions []float64, amplitudes []float64, err error) {
	if len(input) < 2 {
		return nil, nil, fmt.Errorf("input length should be >= 2")
	}

	scale := pd.params.Range / float64(len(input)-1)
	peaks := make([]peak, 0, len(input))

	i := max(0, int(math.Ceil(pd.params.MinPosition/scale)))

	if i+1 < len(input) && input[i] > input[i+1] && input[i] > pd.params.Threshold {
		peaks = append(peaks, peak{position: float64(i) * scale, magnitude: input[i]})
	}

	for {
		for i+1 < len(input)-1 && input[i] >= input[i+1] {
			i++
		}
		for i+1 < len(input)-1 && input[i] < input[i+1] {
			i++
		}

		j := i
		for j+1 < len(input)-1 && input[j] == input[j+1] {
			j++
		}

		if j+1 < len(input)-1 && input[j+1] < input[j] && input[j] > pd.params.Threshold {
			resultVal, resultBin := 0.0, 0.0

			if j != i {
				resultVal = input[i]
				if pd.params.ShouldInterpolate {
					resultBin = float64(i+j) * 0.5
				} else {
					resultBin = float64(i)
				}
			} else {
				if pd.params.ShouldInterpolate {
					resultVal, resultBin = interpolate(input[j-1], input[j], input[j+1], j)
				} else {
					resultVal, resultBin = input[j], float64(j)
				}
			}

			resultPos := resultBin * scale
			if resultPos > pd.params.MaxPosition {
				break
			}
			peaks = append(peaks, peak{position: resultPos, magnitude: resultVal})
		}

		i = j

		if i+1 >= len(input)-1 {
			if i == len(input)-2 && input[i-1] < input[i] && input[i+1] < input[i] && input[i] > pd.params.Threshold {
				resultBin, resultVal := 0.0, 0.0
				if pd.params.ShouldInterpolate {
					resultVal, resultBin = interpolate(input[i-1], input[i], input[i+1], i)
				} else {
					resultVal, resultBin = input[i], float64(i)
				}
				peaks = append(peaks, peak{position: resultBin * scale, magnitude: resultVal})
			}
			break
		}
	}

	pos := pd.params.MaxPosition / scale
	if float64(len(input)-2) < pos && pos <= float64(len(input)-1) && input[len(input)-1] > input[len(input)-2] && input[len(input)-1] > pd.params.Threshold {
		peaks = append(peaks, peak{position: float64(len(input)-1) * scale, magnitude: input[len(input)-1]})
	}

	if pd.params.MinPeakDistance > 0 && len(peaks) > 1 {
		sortPeaksByMagnitude(peaks)

		for k := range len(peaks) - 1 {
			deletedPeaks := make([]int, 0, len(peaks))
			minPos := peaks[k].position - pd.params.MinPeakDistance
			maxPos := peaks[k].position + pd.params.MinPeakDistance
			for l := k + 1; l < len(peaks); l++ {
				if peaks[l].position > minPos && peaks[l].position < maxPos {
					deletedPeaks = append(deletedPeaks, l)
				}
			}
			slices.SortFunc(deletedPeaks, func(a, b int) int {
				return cmp.Compare(b, a)
			})
			for _, idx := range deletedPeaks {
				peaks = slices.Delete(peaks, idx, idx+1)
			}
		}

		if pd.params.OrderBy == PeakOrderByPosition {
			sortPeaksByPosition(peaks)
		}
	} else {
		if pd.params.OrderBy == PeakOrderByAmplitude {
			sortPeaksByMagnitude(peaks)
		}
	}

	wantPeaks := min(pd.params.MaxPeaks, len(peaks))
	positions = make([]float64, wantPeaks)
	amplitudes = make([]float64, wantPeaks)

	for i, peak := range peaks[:wantPeaks] {
		positions[i] = peak.position
		amplitudes[i] = peak.magnitude
	}

	return positions, amplitudes, nil
}

/**
* http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html
*
* Estimating the "true" maximum peak (frequency and magnitude) of the detected local maximum
* using a parabolic curve-fitting. The idea is that the main-lobe of spectrum of most analysis
* windows on a dB scale looks like a parabola and therefore the maximum of a parabola fitted
* through a local maximum bin and it's two neighboring bins will give a good approximation
* of the actual frequency and magnitude of a sinusoid in the input signal.
*
* The parabola f(x) = a(x-n)^2 + b(x-n) + c can be completely described using 3 points;
* f(n-1) = A1, f(n) = A2 and f(n+1) = A3, where
* A1 = 20log10(|X(n-1)|), A2 = 20log10(|X(n)|), A3 = 20log10(|X(n+1)|).
*
* Solving these equation yields: a = 1/2*A1 - A2 + 1/2*A3, b = 1/2*A3 - 1/2*A1 and
* c = A2.
*
* As the 3 bins are known to be a maxima, solving d/dx f(x) = 0, yields the fractional bin
* position x of the estimated peak. Substituting delta_x for (x-n) in this equation yields
* the fractional offset in bins from n where the peak's maximum is.
*
* Solving this equation yields: delta_x = 1/2 * (A1 - A3)/(A1 - 2*A2 + A3).
*
* Computing f(n+delta_x) will estimate the peak's magnitude (in dB's):
* f(n+delta_x) = A2 - 1/4*(A1-A3)*delta_x.
 */
func interpolate(leftVal, middleVal, rightVal float64, currentBin int) (resultVal, resultBin float64) {
	deltaX := 0.5 * ((leftVal - rightVal) / (leftVal - 2*middleVal + rightVal))
	resultVal = middleVal - 0.25*(leftVal-rightVal)*deltaX
	resultBin = float64(currentBin) + deltaX
	return
}

// sortPeaksByMagnitude sorts the peaks slice in place by magnitude in descending order.
// If magnitudes are equal, it sorts by position in ascending order.
func sortPeaksByMagnitude(peaks []peak) {
	slices.SortFunc(peaks, func(a, b peak) int {
		if a.magnitude != b.magnitude {
			return cmp.Compare(b.magnitude, a.magnitude)
		}
		return cmp.Compare(a.position, b.position)
	})
}

// sortPeaksByPosition sorts the peaks slice in place by position in ascending order.
// If positions are equal, it sorts by magnitude in descending order.
func sortPeaksByPosition(peaks []peak) {
	slices.SortFunc(peaks, func(a, b peak) int {
		if a.position != b.position {
			return cmp.Compare(a.position, b.position)
		}
		return cmp.Compare(b.magnitude, a.magnitude)
	})
}
