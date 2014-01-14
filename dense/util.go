package dense

import (
	"math"
)

// add returns slice out whose elements are
// element-wise sums of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func add(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), errOutLength)
	for i, v := range x {
		out[i] = v + y[i]
	}
	return out
}

// add_scaled returns slice out whose elements are
// element-wise sums of x and scaled y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func add_scaled(x, y []float64, s float64, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), errOutLength)
	for i, v := range x {
		out[i] = v + y[i]*s
	}
	return out
}

// subtract returns slice out whose elements are
// element-wise differences of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func subtract(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), errOutLength)
	for i, v := range x {
		out[i] = v - y[i]
	}
	return out
}

// multiply returns slice out whose elements are
// element-wise products of x and y.
// If out is nil, a new slice will be created;
// otherwise, len(out) must equal len(x), as well as len(y).
// out can be x or y, that is, the result is written to one of the input
// slices.
func multiply(x, y, out []float64) []float64 {
	if len(x) != len(y) {
		panic("input length mismatch")
	}
	out = use_slice(out, len(x), errOutLength)
	for i, v := range x {
		out[i] = v * y[i]
	}
	return out
}

func dot(x, y []float64) float64 {
	if len(x) != len(y) {
		panic(errLength)
	}
	d := 0.0
	for i, v := range x {
		d += v * y[i]
	}
	return d
}

// This signature is made to be consistent with shift and scale.
func fill(_ []float64, v float64, out []float64) []float64 {
	for i := range out {
		out[i] = v
	}
	return out
}

/*
func fill(_ []float64, v float64, out []float64) []float64 {
	out[0] = v
	for i, n := 1, len(out); i < n; {
		i += copy(out[i:], out[:i])
	}
}
*/

func zero(x []float64) {
	fill(nil, 0.0, x)
}

// shift adds constant v to every element of x,
// store the result in out and returns out.
// If out is nil, a new slice will be allocated;
// otherwise, out must have the same length as x.
// out can be x itself, in which case elements
// of x are incremented by the amount v.
func shift(x []float64, v float64, out []float64) []float64 {
	out = use_slice(out, len(x), errOutLength)
	for i, val := range x {
		out[i] = val + v
	}
	return out
}

// scale multiplies constant v to every element of x,
// store the result in out and returns out.
// If out is nil, a new slice will be allocated;
// otherwise, out must have the same length as x.
// out can be x itself, in which case elements
// of x are scaled by the amount v.
func scale(x []float64, v float64, out []float64) []float64 {
	out = use_slice(out, len(x), errOutLength)
	for i, val := range x {
		out[i] = val * v
	}
	return out
}

func min(x []float64) float64 {
	v := x[0]
	for _, val := range x {
		if val < v {
			v = val
		}
	}
	return v
}

func max(x []float64) float64 {
	v := x[0]
	for _, val := range x {
		if val > v {
			v = val
		}
	}
	return v
}

func sum(x []float64) float64 {
	v := 0.0
	for _, val := range x {
		v += val
	}
	return v
}

func equal(x, y []float64) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xx := range x {
		if xx != y[i] {
			return false
		}
	}
	return true
}

func equal_approx(x, y []float64, eps float64) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xx := range x {
		if math.Abs(xx-y[i]) > eps {
			return false
		}
	}
	return true
}

// Norm returns the L norm of the slice S, defined as
// (sum_{i=1}^N s[i]^N)^{1/N}
// Special cases:
// L = math.Inf(1) gives the maximum value
// Does not correctly compute the zero norm (use Count).
func norm(s []float64, L float64) (res float64) {
	// Should this complain if L is not positive?
	// Should this be done in log space for better numerical stability?
	//	would be more cost
	//	maybe only if L is high?
	switch {
	case L == 2:
		for _, val := range s {
			res += val * val
		}
		res = math.Pow(res, 0.5)
	case L == 1:
		for _, val := range s {
			res += math.Abs(val)
		}
	case math.IsInf(L, 1):
		res = max(s)
	default:
		for _, val := range s {
			res += math.Pow(math.Abs(val), L)
		}
		res = math.Pow(res, 1.0/L)
	}
	return res
}

func smaller(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func larger(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// use_slice takes a slice x and required length,
// returns x if it is of correct length,
// returns a newly created slice if x is nil,
// and panic if x is non-nil but has wrong length.
func use_slice(x []float64, n int, err error) []float64 {
	if x == nil {
		return make([]float64, n)
	}
	if len(x) != n {
		panic(err)
	}
	return x
}

// use_dense takes a Dense x and required shape,
// returns x if it is of correct shape,
// returns a newly created Dense if x is nil,
// and panic if x is non-nil but has wrong shape.
func use_dense(x *Dense, r, c int, err error) *Dense {
	if x == nil {
		return NewDense(r, c)
	}
	m, n := x.Dims()
	if m != r || n != c {
		panic(err)
	}
	return x
}

func eye(k int) *Dense {
	x := NewDense(k, k)
	x.FillDiag(1.0)
	return x
}

// A Panicker is a function that may panic.
type Panicker func()

// Maybe will recover a panic with a type dense.err from fn, and return this error.
// Any other error is re-panicked.
func Maybe(fn Panicker) (e error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			if e, ok = r.(err); ok {
				return
			}
			panic(r)
		}
	}()
	fn()
	return
}

// A FloatPanicker is a function that returns a float64 and may panic.
type FloatPanicker func() float64

// MaybeFloat will recover a panic with a type dense.err from fn, and return this error.
// Any other error is re-panicked.
func MaybeFloat(fn FloatPanicker) (f float64, e error) {
	defer func() {
		if r := recover(); r != nil {
			if er, ok := r.(err); ok {
				e = er
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// Must can be used to wrap a function returning an error.
// If the returned error is not nil, Must will panic.
func Must(e error) {
	if e != nil {
		panic(e)
	}
}

// Type err represents dense package errors.
// These errors can be recovered by Maybe wrappers.
type err string

func (e err) Error() string { return "dense: " + string(e) }

const (
	errIndexOutOfRange = err("index out of range")
	errZeroLength      = err("zero length in matrix definition")
	errRowLength       = err("row length mismatch")
	errColLength       = err("col length mismatch")
	errSquare          = err("expect square matrix")
	errNormOrder       = err("invalid norm order for matrix")
	errSingular        = err("matrix is singular")
	errLength          = err("length mismatch")
	errShape           = err("dimension mismatch")
	errIllegalStride   = err("illegal stride")
	errPivot           = err("malformed pivot list")
	errIllegalOrder    = err("illegal order")
	errNoEngine        = err("no blas engine registered: call Register()")
	errInLength        = err("input data has wrong length")
	errOutLength       = err("output receiving slice has wrong length")
	errOutShape        = err("output receiving matrix has wrong shape")
)
