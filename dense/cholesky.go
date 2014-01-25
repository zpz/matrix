// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the CholeskyDecomposition class from Jama 1.0.3.

package dense

import (
	"math"
)

// CholFactors contains the Cholesky factors of a symmetric, positive
// definite matrix M.
// The member matrix l is lower triangular (including diagonal)
// that satisfies l * l' = M.
type CholFactors struct {
	l *Dense
}

// Chol returns the Cholesky decomposition of the matrix M.
func Chol(M *Dense) (*CholFactors, bool) {
	ch := &CholFactors{nil}
	n := M.Rows()
	if M.Cols() != n {
		return ch, false
	}
	b := ch.Chol(M)
	return ch, b
}

// Chol conducts Cholesky decomposition for the matrix M.
// The receiver ch is updated. Success flag is returned.
func (ch *CholFactors) Chol(M *Dense) bool {
	n := M.Rows()
	if M.Cols() != n {
		ch.l = nil
		return false
	}

	if ch.l == nil || ch.l.Rows() < n {
		ch.l = NewDense(n, n)
	} else {
		if ch.l.Rows() > n {
			ch.l = ch.l.SubmatrixView(0, 0, n, n)
		}
		ch.l.FillUpper(0.0)
	}

	l := ch.l

	// Typically Chol is called when the caller knows that
	// M is symmetric and PD in concept, e.g. M is a covariance matrix.
	// Hence symmetry is not checked directly.

	for i := 0; i < n; i++ {
		var d float64
		lRowi := l.RowView(i)
		for k := 0; k < i; k++ {
			lRowk := l.RowView(k)
			s := dot(lRowk[:k], lRowi[:k])
			s = (M.Get(i, k) - s) / lRowk[k]
			lRowi[k] = s
			d += s * s
		}
		d = M.Get(i, i) - d
		if d <= 0 {
			ch.l = nil
			return false
		}
		lRowi[i] = math.Sqrt(d)
	}

	return true
}

// L returns the Cholesky factor L such that
// L * L' = M, where M is the original matrix
// that produced ch. Since the returned matrix is
// a reference to internal data of ch, one is
// not expected to make changes to it.
func (ch *CholFactors) L() *Dense {
	return ch.l
}

// Solve returns a matrix x that solves a * x = b where a is the matrix
// that produced ch by Chol(a).
// The matrix b must have the same number of rows as a.
// b is overwritten by the operation and returned containing the
// solution.
func (ch *CholFactors) Solve(b *Dense) *Dense {
	l := ch.l
	if l == nil {
		panic(errInNil)
	}

	n := l.Rows()

	if b.Rows() != n {
		panic(errShapes)
	}

	x := b
	nx := x.Cols()

	// Solve L*Y = B;
	for row := 0; row < n; row++ {
		lrow := l.RowView(row)
		for col := 0; col < nx; col++ {
			ix := col
			v := 0.0
			for k := 0; k < row; k++ {
				v += x.data[ix] * lrow[k]
				ix += x.stride
			}

			// element (row, col) of x.
			x.data[ix] = (x.data[ix] - v) / lrow[row]
		}
	}

	// Solve L'*X = Y;
	for row := n - 1; row >= 0; row-- {
		for col := 0; col < nx; col++ {
			ix := col + x.stride*(n-1)
			// The col-th col of x.
			il := row + l.stride*(n-1)
			v := 0.0
			for k := n - 1; k > row; k-- {
				v += x.data[ix] * l.data[il]
				ix -= x.stride
				il -= l.stride
			}

			// element (row, col) of x.
			x.data[ix] = (x.data[ix] - v) / l.data[il]
		}
	}

	return x
}

// SolveR returns a matrix x that solves x * a = b where a is the matrix
// that produced ch by CholR(a).
// The matrix b must have the same number of cols as a.
// b is overwritten by the operation and returned containing the
// solution.
func (ch *CholFactors) SolveR(b *Dense) *Dense {
	l := ch.l
	if l == nil {
		panic(errInNil)
	}

	n := l.Cols()

	if b.Cols() != n {
		panic(errShapes)
	}

	x := b
	nx := x.Rows()

	// x * U' * U = B

	// Solve Y * U = B
	for col := 0; col < n; col++ {
		for row := 0; row < nx; row++ {
			xrow := x.RowView(row)
			lcol := l.RowView(col)
			v := dot(xrow[:col], lcol[:col])
			xrow[col] = (xrow[col] - v) / lcol[col]
		}
	}

	// Solve X * U' = Y
	for col := n - 1; col >= 0; col-- {
		for row := 0; row < nx; row++ {
			xrow := x.RowView(row)
			il := col + l.stride*(n-1)
			v := 0.0
			for k := n - 1; k > col; k-- {
				v += xrow[k] * l.data[il]
				il -= l.stride
			}
			xrow[col] = (xrow[col] - v) / l.data[il]
		}
	}

	return x
}

// Inv returns the inverse of the matrix a that produced ch by Chol(a).
func (ch *CholFactors) Inv(out *Dense) *Dense {
	l := ch.l
	if l == nil {
		panic(errInNil)
	}

	n := l.Rows()

	if out == nil {
		out = NewDense(n, n)
	} else {
		if out.Rows() != n || out.cols != n {
			panic(errOutShape)
		}
		out.Fill(0.0)
	}
	out.FillDiag(1.0)

	return ch.Solve(out)
}
