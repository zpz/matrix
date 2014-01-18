// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the LUDecomposition class from Jama 1.0.3.

package dense

import (
	"math"
)

type LUFactors struct {
	lu    *Dense
	pivot []int
	sign  int
}

// LU performs an LU decomposition for an m-by-n matrix a.
//
// If m >= n, the LU decomposition is an m-by-n unit lower triangular matrix L,
// an n-by-n upper triangular matrix U, and a permutation vector piv of length m
// so that A(piv,:) = L*U.
//
// If m < n, then L is m-by-m and U is m-by-n.
//
// The LU decompostion with pivoting always exists, even if the matrix is
// singular, so LU will never fail. The primary use of the LU decomposition
// is in the solution of square systems of simultaneous linear equations.  This
// will fail if IsSingular() returns true.
//
// The input matrix a is modified in place and contained in the output.
// If this is not desired, pass in a clone of the source matrix as a.
//
// Use a "left-looking", dot-product, Crout/Doolittle algorithm.
func LU(a *Dense) LUFactors {
	lu := a
	m, n := lu.Dims()

	piv := make([]int, m)
	for i := range piv {
		piv[i] = i
	}
	sign := 1

	luColj := make([]float64, m)

	// Outer loop.
	for j := 0; j < n; j++ {

		// Make a copy of the j-th column to localize references.
		lu.GetCol(j, luColj)

		// Apply previous transformations.
		for i := 0; i < m; i++ {
			luRowi := lu.RowView(i)

			// Most of the time is spent in the following dot product.
			kmax := smaller(i, j)
			s := dot(luRowi[:kmax], luColj[:kmax])

			luColj[i] -= s
			luRowi[j] = luColj[i]
		}

		// Find pivot and exchange if necessary.
		p := j
		for i := j + 1; i < m; i++ {
			if math.Abs(luColj[i]) > math.Abs(luColj[p]) {
				p = i
			}
		}
		if p != j {
			swap(lu.RowView(p), lu.RowView(j))
			piv[p], piv[j] = piv[j], piv[p]
			sign = -sign
		}

		// Compute multipliers.
		if v := lu.Get(j, j); j < m && v != 0 {
			for i := j + 1; i < m; i++ {
				lu.Set(i, j, lu.Get(i, j)/v)
			}
		}
	}

	return LUFactors{lu, piv, sign}
}

// LUGaussian performs an LU Decomposition for an m-by-n matrix a using Gaussian elimination.
// L and U are found using the "daxpy"-based elimination algorithm used in LINPACK and
// MATLAB.
//
// If m >= n, the LU decomposition is an m-by-n unit lower triangular matrix L,
// an n-by-n upper triangular matrix U, and a permutation vector piv of length m
// so that A(piv,:) = L*U.
//
// If m < n, then L is m-by-m and U is m-by-n.
//
// The LU decompostion with pivoting always exists, even if the matrix is
// singular, so the LUD will never fail. The primary use of the LU decomposition
// is in the solution of square systems of simultaneous linear equations.  This
// will fail if IsSingular() returns true.
//
// The input matrix a is modified in place and contained in the output.
// If this is not desired, pass in a clone of the source matrix as a.
func LUGaussian(a *Dense) LUFactors {
	// Initialize.
	m, n := a.Dims()
	lu := a

	piv := make([]int, m)
	for i := range piv {
		piv[i] = i
	}
	sign := 1

	// Main loop.
	for k := 0; k < n; k++ {
		// Find pivot.
		p := k
		for i := k + 1; i < m; i++ {
			if math.Abs(lu.Get(i, k)) > math.Abs(lu.Get(p, k)) {
				p = i
			}
		}

		// Exchange if necessary.
		if p != k {
			swap(lu.RowView(p), lu.RowView(k))
			piv[p], piv[k] = piv[k], piv[p]
			sign = -sign
		}

		// Compute multipliers and eliminate k-th column.
		if lu.Get(k, k) != 0 {
			rowk := lu.RowView(k)
			for i := k + 1; i < m; i++ {
				rowi := lu.RowView(i)
				vik := rowi[k] / rowk[k]
				rowi[k] = vik
				add_scaled(rowi[k+1:], rowk[k+1:], -vik, rowi[k+1:])
			}
		}
	}

	return LUFactors{lu, piv, sign}
}

// IsSingular returns whether the the upper triangular factor and hence a is
// singular.
func (f LUFactors) IsSingular() bool {
	_, n := f.lu.Dims()
	for j := 0; j < n; j++ {
		if f.lu.Get(j, j) == 0 {
			return true
		}
	}
	return false
}

// L returns the lower triangular factor of the LU decomposition.
func (f LUFactors) L() *Dense {
	m, n := f.lu.Dims()
	l := NewDense(m, n)
	if m == n {
		CopyLower(l, f.lu)
		l.FillDiag(1)
	} else {
		k := smaller(m, n)
		CopyLower(l.SubmatrixView(0, 0, k, k), f.lu.SubmatrixView(0, 0, k, k))
		l.SubmatrixView(0, 0, k, k).FillDiag(1)
		if m > n {
			Copy(l.SubmatrixView(n, 0, m-n, n), f.lu.SubmatrixView(n, 0, m-n, n))
		}
	}

	return l
}

// U returns the upper triangular factor of the LU decomposition.
func (f LUFactors) U() *Dense {
	m, n := f.lu.Dims()
	u := NewDense(m, n)
	if m == n {
		CopyUpper(u, f.lu)
		CopyDiag(u, f.lu)
	} else {
		k := smaller(m, n)
		CopyUpper(u.SubmatrixView(0, 0, k, k), f.lu.SubmatrixView(0, 0, k, k))
		CopyDiag(u.SubmatrixView(0, 0, k, k), f.lu.SubmatrixView(0, 0, k, k))
		if n > m {
			Copy(u.SubmatrixView(0, m, m, n-m), f.lu.SubmatrixView(0, m, m, n-m))
		}
	}
	return u
}

// Det returns the determinant of matrix a decomposed into lu. The matrix
// a must have been square.
func (f LUFactors) Det() float64 {
	m, n := f.lu.Dims()
	if m != n {
		panic(errSquare)
	}

	// Product of diagonal elements.
	d := float64(f.sign)
	for j, k := 0, 0; j < n; j++ {
		d *= f.lu.data[k]
		k += f.lu.stride + 1
	}
	return d
}

// Solve computes a solution of a.x = b where b has as many rows as a. A matrix x
// is returned that minimizes the two norm of L*U*X = B(piv,:). QRSolve will panic
// if a is singular. The matrix b is overwritten during the call, and is
// returned.
func (f LUFactors) Solve(b *Dense) *Dense {
	m, n := f.lu.Dims()
	if b.Rows() != m {
		panic(errShapes)
	}
	if f.IsSingular() {
		panic(errSingular)
	}

	// Copy right hand side with pivoting
	pivotRows(b, f.pivot)

	// Solve L*Y = B(piv,:)
	for k := 0; k < n; k++ {
		for i := k + 1; i < n; i++ {
			add_scaled(b.RowView(i), b.RowView(k),
				-f.lu.Get(i, k), b.RowView(i))
		}
	}

	// Solve U*X = Y;
	for k := n - 1; k >= 0; k-- {
		scale(b.RowView(k), 1./f.lu.Get(k, k), b.RowView(k))
		for i := 0; i < k; i++ {
			add_scaled(b.RowView(i), b.RowView(k), -f.lu.Get(i, k),
				b.RowView(i))
		}
	}

	return b
}

func pivotRows(a *Dense, piv []int) *Dense {
	visit := make([]bool, len(piv))
	for to, from := range piv {
		for to != from && !visit[from] {
			visit[from], visit[to] = true, true
			swap(a.RowView(from), a.RowView(to))
			to, from = from, piv[from]
		}
	}
	return a
}
