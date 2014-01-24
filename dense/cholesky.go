// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the CholeskyDecomposition class from Jama 1.0.3.

package dense

import (
	"math"
)

// Cholesky is the left Cholesky factor of a symmetric, positive
// definite matrix M.
// The member matrix L is lower triangular such that
// L * L' equals M.
type Cholesky struct {
	L *Dense
}

// CholeskyR is the right Cholesky factor of a symmetric, positive
// definite matrix M.
// The member matrix U is upper triangular such that
// U' * U equals M.
type CholeskyR struct {
	U *Dense
}

// Chol returns the left Cholesky decomposition of the matrix M.
func Chol(M *Dense) (*Cholesky, bool) {
	ch := &Cholesky{}
	b := ch.Chol(M)
	return ch, b
}

// Chol conducts left Cholesky decomposition for the matrix M.
// The receiver ch is updated. Success flag is returned.
func (ch *Cholesky) Chol(M *Dense) bool {
	n := M.Rows()
	if M.Cols() != n {
		return false
	}

	if ch.L == nil || ch.L.Rows() < n {
		ch.L = NewDense(n, n)
	} else if ch.L.Rows() > n {
		ch.L = ch.L.SubmatrixView(0, 0, n, n)
	}

	l := ch.L

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
			return false
		}
		lRowi[i] = math.Sqrt(d)
	}

	return true
}

// CholR returns the right Cholesky decomposition of the matrix M.
func CholR(M *Dense) (*CholeskyR, bool) {
	ch := &CholeskyR{}
	b := ch.Chol(M)
	return ch, b
}

// Chol conducts right Cholesky decomposition for the matrix M.
// The receiver ch is updated. Success flag is returned.
func (ch *CholeskyR) Chol(M *Dense) bool {
	n := M.Rows()
	if M.Cols() != n {
		return false
	}

	if ch.U == nil || ch.U.Rows() < n {
		ch.U = NewDense(n, n)
	} else if ch.U.Rows() > n {
		ch.U = ch.U.SubmatrixView(0, 0, n, n)
	}

	u := ch.U

	for col := 0; col < n; col++ {
		var d float64
		for k := 0; k < col; k++ {
			s := M.Get(k, col)
			for i := 0; i < k; i++ {
				s -= u.Get(i, k) * u.Get(i, col)
				// TODO: some improvement is possible here,
				// noticing that part of the dot product is repeated.
			}
			s /= u.Get(k, k)
			u.Set(k, col, s)
			d += s * s
		}
		d = M.Get(col, col) - d
		if d <= 0 {
			return false
		}
		u.Set(col, col, math.Sqrt(d))
	}

	return true
}

// Solve returns a matrix x that solves a * x = b where a is the matrix
// that produced ch by Chol(a).
// The matrix b must have the same number of rows as a.
// b is overwritten by the operation and returned containing the
// solution.
func (ch *Cholesky) Solve(b *Dense) *Dense {
	l := ch.L
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

// Solve returns a matrix x that solves x * a = b where a is the matrix
// that produced ch by CholR(a).
// The matrix b must have the same number of cols as a.
// b is overwritten by the operation and returned containing the
// solution.
func (ch *CholeskyR) Solve(b *Dense) *Dense {
	u := ch.U
	if u == nil {
		panic(errInNil)
	}

	n := u.Cols()

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
			iu := col
			v := 0.0
			for k := 0; k < col; k++ {
				v += xrow[k] * u.data[iu]
				iu += u.stride
			}
			xrow[col] = (xrow[col] - v) / u.data[iu]
		}
	}

	// Solve X * U' = Y
	for col := n - 1; col >= 0; col-- {
		ucol := u.RowView(col)
		for row := 0; row < nx; row++ {
			xrow := x.RowView(row)
			v := dot(xrow[col+1:], ucol[col+1:])
			xrow[col] = (xrow[col] - v) / ucol[col]
		}
	}

	return x
}

// Inv returns the inverse of the matrix a that produced ch by Chol(a).
func (ch *Cholesky) Inv(out *Dense) *Dense {
	l := ch.L
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

// Inv returns the inverse of the matrix a that produced ch by CholR(a).
func (ch *CholeskyR) Inv(out *Dense) *Dense {
	u := ch.U
	if u == nil {
		panic(errInNil)
	}

	n := u.Rows()

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
