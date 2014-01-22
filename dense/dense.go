// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dense

import (
	"github.com/gonum/blas"
	"math"
)

var blasEngine blas.Float64

func Register(b blas.Float64) { blasEngine = b }

// This package uses row-major storage.
// Every operation is affected by it.
// Do not change it.
const blasOrder = blas.RowMajor

type Dense struct {
	rows, cols, stride int
	data               []float64
}

// NewDense creates a Dense with r rows and c cols,
// allocates memory to hold all elements of the matrix,
// and return the newly-created, all-zero matrix.
func NewDense(r, c int) *Dense {
	return &Dense{
		rows:   r,
		cols:   c,
		stride: c,
		data:   make([]float64, r*c),
	}
}

// DenseView creates a view into the slice data as a Dense with
// r rows and c cols. If the slice data is assigned before being passed
// to this function, the slice and the created Dense become a view to
// each other: changes to the elements via one of the two are reflected
// in the other.
// If data is created on-the-fly in a call to this function, then the
// created Dense is no different from one created using NewDense in the
// sense that the data is internal to the Dense.
// This function does not allocate new memory.
func DenseView(data []float64, r, c int) *Dense {
	if len(data) != r*c {
		panic(errInLength)
	}
	var m Dense
	m.rows = r
	m.cols = c
	m.stride = c
	m.data = data
	return &m
}

func (m *Dense) Dims() (r, c int) { return m.rows, m.cols }

func (m *Dense) Rows() int { return m.rows }

func (m *Dense) Cols() int { return m.cols }

// Contiguous reports whether the data of the matrix is stored
// in a contiguous segment of a slice.
// The returned value is false if and only if the matrix is
// the submatrix view of another matrix and has fewer columns
// than its parent matrix; otherwise, the value is true.
// If this function returns true, one may subsequently
// call DataView to get a view of the data slice and work on it directly.
func (m *Dense) Contiguous() bool { return m.cols == m.stride }

func (m *Dense) Get(r, c int) float64 {
	return m.data[r*m.stride+c]
}

func (m *Dense) Set(r, c int, v float64) *Dense {
	m.data[r*m.stride+c] = v
	return m
}

func (m *Dense) RowView(r int) []float64 {
	if r >= m.rows || r < 0 {
		panic(errIndexOutOfRange)
	}
	k := r * m.stride
	return m.data[k : k+m.cols]
}

func (m *Dense) GetRow(r int, row []float64) []float64 {
	row = use_slice(row, m.cols, errOutLength)
	copy(row, m.RowView(r))
	return row
}

func (m *Dense) SetRow(r int, v []float64) *Dense {
	if len(v) != m.cols {
		panic(errInLength)
	}
	copy(m.RowView(r), v)
	return m
}

// ColView
// There is no ColView b/c of row-major.

func (m *Dense) GetCol(c int, col []float64) []float64 {
	if c >= m.cols || c < 0 {
		panic(errIndexOutOfRange)
	}
	col = use_slice(col, m.rows, errOutLength)

	if blasEngine == nil {
		panic(errNoEngine)
	}
	blasEngine.Dcopy(m.rows, m.data[c:], m.stride, col, 1)

	return col
}

func (m *Dense) SetCol(c int, v []float64) *Dense {
	if c >= m.cols || c < 0 {
		panic(errIndexOutOfRange)
	}

	if len(v) != m.rows {
		panic(errInLength)
	}

	if blasEngine == nil {
		panic(errNoEngine)
	}
	blasEngine.Dcopy(m.rows, v, 1, m.data[c:], m.stride)
	return m
}

// SubmatrixView returns a "view" to the specified sub-matrix.
// Changes made to the elements of this view-matrix are reflected in the
// original matrix, and vice versa. This function does not allocate new memory,
// because the view points to the data of the original matrix.
func (m *Dense) SubmatrixView(i, j, r, c int) *Dense {
	if i < 0 || i >= m.rows || r <= 0 || i+r > m.rows {
		panic(errIndexOutOfRange)
	}
	if j < 0 || j >= m.cols || c <= 0 || j+c > m.cols {
		panic(errIndexOutOfRange)
	}

	out := Dense{}
	out.rows = r
	out.cols = c
	out.stride = m.stride
	out.data = m.data[i*m.stride+j : (i+r-1)*m.stride+(j+c)]
	return &out
}

// GetSubmatrix copies all elements in the specified submatrix
// into slice out, row by row. If out is nil, a new slice is created;
// otherwise, out must have the correct length, and it is written into.
// The stuffed slice is returned.
// Note that the output is a slice rather than a matrix.
// To copy out the submatrix into a matrix, use
//    Copy(dest, src.SubmatrixView(...))
// or
//    Clone(src.SubmatrixView(...))
func (m *Dense) GetSubmatrix(i, j, r, c int, out []float64) []float64 {
	return m.SubmatrixView(i, j, r, c).GetData(out)
}

// SetSubmatrix copies values in slice v to the specified submatrix,
// row by row. The receiver matrix is updated in-place, and is also
// returned.
// Note that the source is a slice rather than a matrix.
// To copy into the submatrix from a matrix, use
//    Copy(dest.SubmatrixView(...), src)
func (m *Dense) SetSubmatrix(i, j, r, c int, v []float64) *Dense {
	m.SubmatrixView(i, j, r, c).SetData(v)
	// Note: do not 'return' this line; that would return the
	// submatrix, not m.
	return m
}

// DataView returns the slice in the matrix object
// that holds the data, in row major.
// Subsequent changes to the returned slice is reflected
// in the original matrix, and vice versa.
// This is possible only when Contiguous() is true;
// if Contiguous() is false, nil is returned.
func (m *Dense) DataView() []float64 {
	if m.Contiguous() {
		return m.data
	}
    panic("data is not contiguous")
}

// GetData copies out all elements of the matrix, row by row, in the
// slice out.  If out is nil, a new slice is created; otherwise out must
// have the correct length.
// The copied slice is returned.
func (m *Dense) GetData(out []float64) []float64 {
	out = use_slice(out, m.rows*m.cols, errOutLength)
	if m.Contiguous() {
		copy(out, m.DataView())
	} else {
		r, c := m.rows, m.cols
		for row, k := 0, 0; row < r; row++ {
			copy(out[k:k+c], m.RowView(row))
			k += c
		}
	}
	return out
}

// SetData copies the values of v into the matrix.
// Values in v are supposed to be in row major, that is,
// values for the first row of the matrix, followed by
// values for the second row, and so on.
// Length of v must be equal to the total number of elements in the
// matrix.
// The matrix is updated in-place, and is also returned.
func (m *Dense) SetData(v []float64) *Dense {
	r, c := m.rows, m.cols
	if len(v) != r*c {
		panic(errInLength)
	}
	if m.Contiguous() {
		copy(m.DataView(), v)
	} else {
		for k, row := 0, 0; row < r; row++ {
			copy(m.RowView(row), v[k:k+c])
			k += c
		}
	}
	return m
}

func (m *Dense) Fill(v float64) *Dense {
	element_wise_unary(m, v, m, fill)
	return m
}

// GetDiag copies diagonal elements of m into out.
// out must have the correct length.
// m is not required to be square.
func (m *Dense) GetDiag(out []float64) []float64 {
	k := smaller(m.rows, m.cols)
	out = use_slice(out, k, errOutLength)
	for i, j := 0, 0; i < k; i += m.stride + 1 {
		out[j] = m.data[i]
		j++
	}
	return out
}

// SetDiag sets diagonal elements of m to the values in v.
// The length of v must be exactly right.
// m is not required to be square.
func (m *Dense) SetDiag(v []float64) *Dense {
	k := smaller(m.rows, m.cols)
	if len(v) != k {
		panic(errInLength)
	}
	for i, j := 0, 0; i < k; i += m.stride + 1 {
		m.data[i] = v[j]
		j++
	}
	return m
}

// FillDiag sets all diagonal elements to value v.
// The matrix m is not required to be square.
func (m *Dense) FillDiag(v float64) *Dense {
	n := smaller(m.rows, m.cols)
	for row, k := 0, 0; row < n; row++ {
		m.data[k] = v
		k += m.stride + 1
	}
	return m
}

// Copy copies the elements of src into dest.
// dest must have the correct dimensions; it can not be nil.
// To create a new matrix and copy into it, use Clone.
func Copy(dest *Dense, src *Dense) {
	if dest.rows != src.rows || dest.cols != src.cols {
		panic(errShapes)
	}
	if dest.Contiguous() && src.Contiguous() {
		copy(dest.DataView(), src.DataView())
	} else {
		for row := 0; row < src.rows; row++ {
			copy(dest.RowView(row), src.RowView(row))
		}
	}
}

// Clone creates a new Dense and copies the elements of src into it.
// The new Dense is returned.
// Note that while src could be a submatrix of a larger matrix,
// the cloned matrix is always freshly allocated and is its own
// entirety.
// Note the "always allocate a new one" nature of Clone.
// To copy into an existing matrix (including a submatrix),
// use Copy instead.
func Clone(src *Dense) *Dense {
	out := NewDense(src.rows, src.cols)
	Copy(out, src)
	return out
}

// CopyDiag copies the diagonal elements of the matrix src
// to the corresponding locations in dest.
// src and dest must have the same shape;
// howerver, they are not required to be square.
// Off-diagonal elements are not touched.
func CopyDiag(dest, src *Dense) {
	if dest.rows != src.rows || dest.cols != src.cols {
		panic(errShapes)
	}
	for row, kd, ks, k := 0, 0, 0, smaller(src.rows, src.cols); row < k; row++ {
		dest.data[kd] = src.data[ks]
		kd += dest.stride + 1
		ks += src.stride + 1
	}
}

// CopyUpper copies above-diagonal elements in src to corresponding
// locations in dest; on- and below-diagonal elements are not touched.
// dest and src must have the save shape; they are not required to be
// square.
func CopyUpper(dest, src *Dense) {
	if dest.rows != src.rows || dest.cols != src.cols {
		panic(errShapes)
	}
	for row, k := 0, smaller(src.rows, src.cols); row < k-1; row++ {
		copy(dest.RowView(row)[row+1:], src.RowView(row)[row+1:])
	}
	if src.cols > src.rows {
		k := src.rows
		copy(dest.RowView(k - 1)[k:], src.RowView(k - 1)[k:])
	}
}

// CopyLower copies below-diagonal elements in src to corresponding
// locations in dest; on- and above-diagonal elements are not touched.
// dest and src must have the save shape; they are not required to be
// square.
func CopyLower(dest, src *Dense) {
	if dest.rows != src.rows || dest.cols != src.cols {
		panic(errShapes)
	}
	for row := 1; row < src.rows; row++ {
		k := smaller(row, src.cols)
		copy(dest.RowView(row)[:k], src.RowView(row)[:k])
	}
}

// FillLower sets the below-diagonal elements
// of m to value v, and return the modified m.
// m is not required to be square.
func (m *Dense) FillLower(v float64) *Dense {
	for row := 1; row < m.rows; row++ {
		if row < m.cols {
			fill(nil, v, m.RowView(row)[:row])
		} else {
			fill(nil, v, m.RowView(row))
		}
	}
	return m
}

// FillUpper sets the above-diagonal elements
// (or elements to the right of the diagonal)
// of m to value v, and return the modified m.
// m is not required to be square.
func (m *Dense) FillUpper(v float64) *Dense {
	for row, k := 0, smaller(m.rows, m.cols); row < k-1; row++ {
		fill(nil, v, m.RowView(row)[(row+1):])
	}
	if m.cols > m.rows {
		fill(nil, v, m.RowView(m.rows - 1)[m.rows:])
	}
	return m
}

func element_wise_unary(a *Dense, val float64, out *Dense,
	f func(a []float64, val float64, out []float64) []float64) *Dense {

	out = use_dense(out, a.rows, a.cols, errOutShape)
	if a.Contiguous() && out.Contiguous() {
		f(a.DataView(), val, out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
		f(a.RowView(row), val, out.RowView(row))
	}
	return out
}

// Shift adds constant v to every element of m,
// returns the new values in matrix out.
// out may be m itself, amounting to in-place update.
func Shift(m *Dense, v float64, out *Dense) *Dense {
	return element_wise_unary(m, v, out, shift)
}

// Shift adds constant v to each element of m.
// The update is in-place; the updated matrix is also returned.
func (m *Dense) Shift(v float64) *Dense {
	return Shift(m, v, m)
}

// Shift multiplies each element of m by constant v,
// returns the new values in matrix out.
// out may be m itself, amounting to in-place update.
func Scale(m *Dense, v float64, out *Dense) *Dense {
	return element_wise_unary(m, v, out, scale)
}

// Scale multiplies each element of m by constant v.
// The update is in-place; the updated matrix is also returned.
func (m *Dense) Scale(v float64) *Dense {
	return Scale(m, v, m)
}

func element_wise_binary(a, b, out *Dense,
	f func(a, b, out []float64) []float64) *Dense {

	if a.rows != b.rows || a.cols != b.cols {
		panic(errShapes)
	}
	out = use_dense(out, a.rows, a.cols, errOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		f(a.DataView(), b.DataView(), out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
		f(a.RowView(row), b.RowView(row), out.RowView(row))
	}
	return out
}

// Add adds matrices a and b, place the result in out and return out.
// If out is nil, a new matrix is allocated and used, and returned.
// If out is non-nil, it must have the correct shape.
// out may be one of a and b, that is, add a and b and place the result
// in a (or b, depending on which one out is).
func Add(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, add)
}

// Add adds matrix X to the receiver matrix.
// The receiver matrix is modified in-place;
// the modified receiver matrix is also returned.
func (m *Dense) Add(X *Dense) *Dense {
	return Add(m, X, m)
}

// AddScaled adds matrix a and matrix b scaled by constant s,
// and place the result in matrix out, which is returned.
// If out is nil, a new matrix is allocated, used, and returned.
// If out is non-nil, it must have the correct shape.
// out may be one of a and b, that is, one of the input matrices holds
// the result.
func AddScaled(a, b *Dense, s float64, out *Dense) *Dense {
	if a.rows != b.rows || a.cols != b.cols {
		panic(errShapes)
	}
	out = use_dense(out, a.rows, a.cols, errOutShape)
	if a.Contiguous() && b.Contiguous() && out.Contiguous() {
		add_scaled(a.DataView(), b.DataView(), s, out.DataView())
		return out
	}
	for row := 0; row < a.rows; row++ {
		add_scaled(a.RowView(row), b.RowView(row), s, out.RowView(row))
	}
	return out
}

// AddScaled adds X scaled by constant s to the receiver matrix.
// The receiver matrix is updated in-place, and is also returned.
func (m *Dense) AddScaled(X *Dense, s float64) *Dense {
	return AddScaled(m, X, s, m)
}

// Subtract is analogous to Add.
func Subtract(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, subtract)
}

// Subtract is analogous to Add.
func (m *Dense) Subtract(X *Dense) *Dense {
	return Subtract(m, X, m)
}

// Elemult does element-wise multiplication in a way analogous to Add
// and Subtract.
func Elemult(a, b, out *Dense) *Dense {
	return element_wise_binary(a, b, out, multiply)
}

// Elemult does element-wise multiplication in a way analogous to Add
// and Subtract.
func (m *Dense) Elemult(X *Dense) *Dense {
	return Elemult(m, X, m)
}

// Mult multiplies matrices a and b, place the result in out, and return
// out. If out is nil, a new matrix is allocated and used.
// TODO: find out whether out can be one of a and b.
func Mult(a, b, out *Dense) *Dense {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(errShapes)
	}

	out = use_dense(out, ar, bc, errOutShape)

	if blasEngine == nil {
		panic(errNoEngine)
	}
	blasEngine.Dgemm(
		blasOrder,
		blas.NoTrans, blas.NoTrans,
		ar, bc, ac,
		1.,
		a.data, a.stride,
		b.data, b.stride,
		0.,
		out.data, out.stride)

	return out
}

func Dot(a, b *Dense) float64 {
	if a.rows != b.rows || a.cols != b.cols {
		panic(errShapes)
	}
	if a.Contiguous() && b.Contiguous() {
		return dot(a.DataView(), b.DataView())
	}
	d := 0.0
	for row := 0; row < a.rows; row++ {
		d += dot(a.RowView(row), b.RowView(row))
	}
	return d
}

func Hstack(a, b, out *Dense) *Dense {
	if a.rows != b.rows {
		panic(errShapes)
	}
	out = use_dense(out, a.rows, a.cols+b.cols, errOutShape)
	Copy(out.SubmatrixView(0, 0, a.rows, a.cols), a)
	Copy(out.SubmatrixView(0, a.cols, b.rows, b.cols), b)
	return out
}

func Vstack(a, b, out *Dense) *Dense {
	if a.cols != b.cols {
		panic(errShapes)
	}
	out = use_dense(out, a.rows+b.rows, a.cols, errOutShape)
	Copy(out.SubmatrixView(0, 0, a.rows, a.cols), a)
	Copy(out.SubmatrixView(a.rows, 0, b.rows, b.cols), b)
	return out
}

func (m *Dense) Min() float64 {
	if m.Contiguous() {
		return min(m.DataView())
	}
	v := min(m.RowView(0))
	for row := 1; row < m.rows; row++ {
		z := min(m.RowView(row))
		if z < v {
			v = z
		}
	}
	return v
}

func (m *Dense) Max() float64 {
	if m.Contiguous() {
		return max(m.DataView())
	}
	v := max(m.RowView(0))
	for row := 1; row < m.rows; row++ {
		z := max(m.RowView(row))
		if z > v {
			v = z
		}
	}
	return v
}

func (m *Dense) Sum() float64 {
	if m.Contiguous() {
		return sum(m.DataView())
	}
	v := 0.0
	for row := 0; row < m.rows; row++ {
		v += sum(m.RowView(row))
	}
	return v
}

func (m *Dense) Trace() float64 {
	if m.rows != m.cols {
		panic(errSquare)
	}
	var t float64
	for i, n := 0, m.rows*m.cols; i < n; i += m.stride + 1 {
		t += m.data[i]
	}
	return t
}

var inf = math.Inf(1)

const (
	epsilon = 2.2204e-16
	small   = math.SmallestNonzeroFloat64
)

func (m *Dense) Norm(ord float64) float64 {
	var n float64
	switch {
	case ord == 1:
		col := make([]float64, m.rows)
		for i := 0; i < m.cols; i++ {
			var s float64
			for _, e := range m.GetCol(i, col) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case math.IsInf(ord, +1):
		for i := 0; i < m.rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Max(math.Abs(s), n)
		}
	case ord == -1:
		n = math.MaxFloat64
		col := make([]float64, m.rows)
		for i := 0; i < m.cols; i++ {
			var s float64
			for _, e := range m.GetCol(i, col) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case math.IsInf(ord, -1):
		n = math.MaxFloat64
		for i := 0; i < m.rows; i++ {
			var s float64
			for _, e := range m.RowView(i) {
				s += e
			}
			n = math.Min(math.Abs(s), n)
		}
	case ord == 0:
		for i := 0; i < len(m.data); i += m.stride {
			for _, v := range m.data[i : i+m.cols] {
				n += v * v
			}
		}
		return math.Sqrt(n)
	case ord == 2, ord == -2:
		s := SVD(m, epsilon, small, false, false).Sigma
		if ord == 2 {
			return s[0]
		}
		return s[len(s)-1]
	default:
		panic(errNormOrder)
	}

	return n
}

// Apply applies function f to each element of m, place the results in
// out, and return out.
// f takes the row index, col index, and entry value, and outputs a
// value.
// Row and col indices are zero-based.
// If out is nil, a new matrix is allocated and used;
// otherwise out must have the correct shape, that is, the same shape as
// m.
// out may be m itself, amounting to in-place update.
func Apply(
	m *Dense,
	f func(r, c int, v float64) float64,
	out *Dense) *Dense {

	out = use_dense(out, m.rows, m.cols, errOutShape)
	for row := 0; row < m.rows; row++ {
		in_row := m.RowView(row)
		out_row := out.RowView(row)
		for col, z := range in_row {
			out_row[col] = f(row, col, z)
		}
	}
	return out
}

// Apply updates each element of m by calling the function f,
// which takes the row index, col index, and the element's value.
// Row and col indices are zero-based.
// The matrix m is modified in-place, and is also returned.
func (m *Dense) Apply(f func(int, int, float64) float64) *Dense {
	return Apply(m, f, m)
}

// T transposes m, places the result in out, and returns out.
// If out is nil, a new matrix is allocated and used;
// otherwise out must have the correct shape.
// If m is square, out can be m itself.
func T(m, out *Dense) *Dense {
	out = use_dense(out, m.cols, m.rows, errOutShape)
	if m.rows == m.cols {
		for row := 0; row < m.rows; row++ {
			for col := 0; col < row; col++ {
				z := m.Get(row, col)
				out.Set(row, col, m.Get(col, row))
				out.Set(col, row, z)
			}
			out.Set(row, row, m.Get(row, row))
		}
	} else {
		for row := 0; row < m.rows; row++ {
			out.SetCol(row, m.RowView(row))
		}
	}
	return out
}

// T transposes the square receiver matrix in-place,
// and also returns the transposed matrix.
func (m *Dense) T() *Dense {
	if m.rows != m.cols {
		panic(errSquare)
	}
	T(m, m)
	return m
}

func Equal(a, b *Dense) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}
	if a.Contiguous() && b.Contiguous() {
		return equal(a.DataView(), b.DataView())
	}
	for row := 0; row < a.rows; row++ {
		if !equal(a.RowView(row), b.RowView(row)) {
			return false
		}
	}
	return true
}

func EqualApprox(a, b *Dense, eps float64) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}
	if a.Contiguous() && b.Contiguous() {
		return equal_approx(a.DataView(), b.DataView(), eps)
	}
	for row := 0; row < a.rows; row++ {
		if !equal_approx(a.RowView(row), b.RowView(row), eps) {
			return false
		}
	}
	return true
}

// Det returns the determinant of the matrix a.
func (m *Dense) Det() float64 {
	return LU(Clone(m)).Det()
}

// Inv returns the inverse or pseudoinverse of the matrix a.
//
// Within this function, a is modified.
// If this is not desired, pass in a clone of the source matrix.
func Inv(a *Dense, out *Dense) *Dense {
	if out == nil {
		out = eye(a.rows)
	} else {
		if out.rows != a.rows || a.cols != a.rows {
			panic(errOutShape)
		}
		out.Fill(0.0)
		out.FillDiag(1.0)
	}
	return Solve(a, out)
}

// Solve returns a matrix x that satisfies ax = b.
//
// Within this function, both a and b are modified;
// b becomes the returned solution matrix.
// If these modifications are not desired,
// pass in clones of the source matrices of a and b.
func Solve(a, b *Dense) *Dense {
	if a.rows == a.cols {
		b = LU(a).Solve(b)
	} else {
		b = QR(a).Solve(b)
	}
	return b
}
