package dense

import (
	"fmt"
	"math"
	"math/rand"
)

func isUpperTriangular(a *Dense) bool {
	rows, cols := a.Dims()
	for c := 0; c < cols-1; c++ {
		for r := c + 1; r < rows; r++ {
			if math.Abs(a.Get(r, c)) > 1e-14 {
				return false
			}
		}
	}
	return true
}

func isOrthogonal(a *Dense) bool {
	rows, cols := a.Dims()
	col1 := make([]float64, rows)
	col2 := make([]float64, rows)
	for i := 0; i < cols-1; i++ {
		for j := i + 1; j < cols; j++ {
			a.GetCol(i, col1)
			a.GetCol(j, col2)
			dot := dot(col1, col2)
			if math.Abs(dot) > 1e-14 {
				return false
			}
		}
	}
	return true
}

func flatten(f [][]float64) (r, c int, d []float64) {
	for _, r := range f {
		d = append(d, r...)
	}
	return len(f), len(f[0]), d
}

func unflatten(r, c int, d []float64) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

func flatten2dense(f [][]float64) *Dense {
	return make_dense(flatten(f))
}

func make_dense(r, c int, data []float64) *Dense {
	return DenseView(data, r, c)
}

func randDense(size int, rho float64, rnd func() float64) (*Dense, error) {
	if size == 0 {
		return nil, errZeroLength
	}
	d := &Dense{
		rows: size, cols: size, stride: size,
		data: make([]float64, size*size),
	}
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if rand.Float64() < rho {
				d.Set(i, j, rnd())
			}
		}
	}
	return d, nil
}

func print_dense(x *Dense) {
	for row := 0; row < x.rows; row++ {
		fmt.Println(x.RowView(row))
	}
}

// A panicker is a function that may panic.
type panicker func()

// maybe will recover a panic with a type dense.err from fn, and return this error.
// Any other error is re-panicked.
func maybe(fn panicker) (e error) {
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

// A floatPanicker is a function that returns a float64 and may panic.
type floatPanicker func() float64

// maybeFloat will recover a panic with a type dense.err from fn, and return this error.
// Any other error is re-panicked.
func maybeFloat(fn floatPanicker) (f float64, e error) {
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

// must can be used to wrap a function returning an error.
// If the returned error is not nil, Must will panic.
func must(e error) {
	if e != nil {
		panic(e)
	}
}
