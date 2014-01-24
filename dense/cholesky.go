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
// The member matrix LU is symmetric; let its lower- and upper-triangles
// (including diagonal) be L and U, then L * U equals M.
type CholFactors struct {
	LU *Dense
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
		ch.LU = nil
		return false
	}

	if ch.LU == nil || ch.LU.Rows() < n {
		ch.LU = NewDense(n, n)
	} else {
		if ch.LU.Rows() > n {
			ch.LU = ch.LU.SubmatrixView(0, 0, n, n)
		}
	}

	lu := ch.LU

	// Typically Chol is called when the caller knows that
	// M is symmetric and PD in concept, e.g. M is a covariance matrix.
	// Hence symmetry is not checked directly.

	for i := 0; i < n; i++ {
		var d float64
		luRowi := lu.RowView(i)
		for k := 0; k < i; k++ {
			luRowk := lu.RowView(k)
			s := dot(luRowk[:k], luRowi[:k])
			s = (M.Get(i, k) - s) / luRowk[k]
			luRowi[k] = s
			d += s * s
		}
		d = M.Get(i, i) - d
		if d <= 0 {
			ch.LU = nil
			return false
		}
		luRowi[i] = math.Sqrt(d)
	}

	// Fill up the upper triangle.
	for row := 0; row < n-1; row++ {
		r := lu.RowView(row)
		k := row + lu.stride*(row+1)
		for col := row + 1; col < n; col++ {
			r[col] = lu.data[k]
			k += lu.stride
		}
	}

	return true
}

// Solve returns a matrix x that solves a * x = b where a is the matrix
// that produced ch by Chol(a).
// The matrix b must have the same number of rows as a.
// b is overwritten by the operation and returned containing the
// solution.
func (ch *CholFactors) Solve(b *Dense) *Dense {
	lu := ch.LU
	if lu == nil {
		panic(errInNil)
	}

	n := lu.Rows()

	if b.Rows() != n {
		panic(errShapes)
	}

	x := b
	nx := x.Cols()

	// Solve L*Y = B;
	for row := 0; row < n; row++ {
		lurow := lu.RowView(row)
		for col := 0; col < nx; col++ {
			ix := col
			v := 0.0
			for k := 0; k < row; k++ {
				v += x.data[ix] * lurow[k]
				ix += x.stride
			}

			// element (row, col) of x.
			x.data[ix] = (x.data[ix] - v) / lurow[row]
		}
	}

	// Solve L'*X = Y;
	for row := n - 1; row >= 0; row-- {
		for col := 0; col < nx; col++ {
			lucol := lu.RowView(row) // the row-th col of lu
			ix := col + x.stride*(n-1)
			v := 0.0
			for k := n - 1; k > row; k-- {
				v += x.data[ix] * lucol[k]
				ix -= x.stride
			}

			// element (row, col) of x.
			x.data[ix] = (x.data[ix] - v) / lucol[row]
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
	lu := ch.LU
	if lu == nil {
		panic(errInNil)
	}

	n := lu.Cols()

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
			lucol := lu.RowView(col)
			v := dot(xrow[:col], lucol[:col])
			xrow[col] = (xrow[col] - v) / lucol[col]
		}
	}

	// Solve X * U' = Y
	for col := n - 1; col >= 0; col-- {
		lucol := lu.RowView(col)
		for row := 0; row < nx; row++ {
			xrow := x.RowView(row)
			v := dot(xrow[col+1:], lucol[col+1:])
			xrow[col] = (xrow[col] - v) / lucol[col]
		}
	}

	return x
}

// Inv returns the inverse of the matrix a that produced ch by Chol(a).
func (ch *CholFactors) Inv(out *Dense) *Dense {
	lu := ch.LU
	if lu == nil {
		panic(errInNil)
	}

	n := lu.Rows()

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
