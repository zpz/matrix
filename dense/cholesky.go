// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the CholeskyDecomposition class from Jama 1.0.3.

package dense

import (
	"math"
)

type CholeskyFactor struct {
	L   *Dense
	SPD bool
}

// CholeskyL returns the left Cholesky decomposition of the matrix a and whether
// the matrix is symmetric or positive definite, the returned matrix l is a lower
// triangular matrix such that a = l.l'.
func Cholesky(a *Dense) CholeskyFactor {
	// Initialize.
	m, n := a.Dims()
	spd := m == n
	l := NewDense(n, n)

	// Main loop.
	for j := 0; j < n; j++ {
		var d float64
		lRowj := l.RowView(j)
		for k := 0; k < j; k++ {
			var s float64
			lRowk := l.RowView(k)
			for i := 0; i < k; i++ {
				s += lRowk[i] * lRowj[i]
			}
			s = (a.Get(j, k) - s) / l.Get(k, k)
			lRowj[k] = s
			d += s * s
			spd = spd && a.Get(k, j) == a.Get(j, k)
		}
		d = a.Get(j, j) - d
		spd = spd && d > 0
		l.Set(j, j, math.Sqrt(math.Max(d, 0)))
		for k := j + 1; k < n; k++ {
			l.Set(j, k, 0)
		}
	}

	return CholeskyFactor{L: l, SPD: spd}
}

// CholeskyR returns the right Cholesky decomposition of the matrix a and whether
// the matrix is symmetric or positive definite, the returned matrix r is an upper
// triangular matrix such that a = r'.r.
func CholeskyR(a *Dense) (r *Dense, spd bool) {
	// Initialize.
	m, n := a.Dims()
	spd = m == n
	r = NewDense(n, n)

	// Main loop.
	for j := 0; j < n; j++ {
		var d float64
		for k := 0; k < j; k++ {
			s := a.Get(k, j)
			for i := 0; i < k; i++ {
				s -= r.Get(i, k) * r.Get(i, j)
			}
			s /= r.Get(k, k)
			r.Set(k, j, s)
			d += s * s
			spd = spd && a.Get(k, j) == a.Get(j, k)
		}
		d = a.Get(j, j) - d
		spd = spd && d > 0
		r.Set(j, j, math.Sqrt(math.Max(d, 0)))
		for k := j + 1; k < n; k++ {
			r.Set(k, j, 0)
		}
	}

	return r, spd
}

// CholeskySolve returns a matrix x that solves a.x = b where a = l.l'. The matrix b must
// have the same number of rows as a, and a must be symmetric and positive definite. The
// matrix b is overwritten by the operation.
func (f CholeskyFactor) Solve(b *Dense) (x *Dense) {
	if !f.SPD {
		panic("mat64: matrix not symmetric positive definite")
	}
	l := f.L

	_, n := l.Dims()
	_, bn := b.Dims()
	if n != bn {
		panic(errShape)
	}

	nx := bn
	x = b

	// Solve L*Y = B;
	for k := 0; k < n; k++ {
		for j := 0; j < nx; j++ {
			for i := 0; i < k; i++ {
				x.Set(k, j, x.Get(k, j)-x.Get(i, j)*l.Get(k, i))
			}
			x.Set(k, j, x.Get(k, j)/l.Get(k, k))
		}
	}

	// Solve L'*X = Y;
	for k := n - 1; k >= 0; k-- {
		for j := 0; j < nx; j++ {
			for i := k + 1; i < n; i++ {
				x.Set(k, j, x.Get(k, j)-x.Get(i, j)*l.Get(i, k))
			}
			x.Set(k, j, x.Get(k, j)/l.Get(k, k))
		}
	}

	return x
}