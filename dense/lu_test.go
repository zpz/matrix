// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dense

import (
	check "launchpad.net/gocheck"
)

func (s *S) TestLUD(c *check.C) {
	for _, t := range []struct {
		a *Dense

		l *Dense
		u *Dense

		pivot []int
		sign  int
	}{
		{ // This is a hard coded equivalent of the approach used in the Jama LU test.
			a: make_dense(3, 3, []float64{
				0, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),

			l: make_dense(3, 3, []float64{
				1, 0, 0,
				0, 1, 0,
				0.5714285714285714, 0.2142857142857144, 1,
			}),
			u: make_dense(3, 3, []float64{
				7, 8, 9,
				0, 2, 3,
				0, 0, 0.2142857142857144,
			}),
			pivot: []int{
				2, // 0 0 1
				0, // 1 0 0
				1, // 0 1 0
			},
			sign: 1,
		},
	} {
		lf := LU(Clone(t.a))
		if t.pivot != nil {
			c.Check(lf.pivot, check.DeepEquals, t.pivot)
			c.Check(lf.sign, check.Equals, t.sign)
		}

		l := lf.L()
		if t.l != nil {
			c.Check(Equal(l, t.l), check.Equals, true)
		}
		u := lf.U()
		if t.u != nil {
			c.Check(Equal(u, t.u), check.Equals, true)
		}

		l = Mult(l, u, nil)
		c.Check(Approx(l, pivotRows(Clone(t.a), lf.pivot), 1e-12), check.Equals, true)

		x := lf.Solve(eye(3))
		t.a = Mult(t.a, x, nil)
		c.Check(Approx(t.a, eye(3), 1e-12), check.Equals, true)
	}
}

func (s *S) TestLUDGaussian(c *check.C) {
	for _, t := range []struct {
		a *Dense

		l *Dense
		u *Dense

		pivot []int
		sign  int
	}{
		{ // This is a hard coded equivalent of the approach used in the Jama LU test.
			a: make_dense(3, 3, []float64{
				0, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),

			l: make_dense(3, 3, []float64{
				1, 0, 0,
				0, 1, 0,
				0.5714285714285714, 0.2142857142857144, 1,
			}),
			u: make_dense(3, 3, []float64{
				7, 8, 9,
				0, 2, 3,
				0, 0, 0.2142857142857144,
			}),
			pivot: []int{
				2, // 0 0 1
				0, // 1 0 0
				1, // 0 1 0
			},
			sign: 1,
		},
	} {
		lf := LUGaussian(Clone(t.a))
		if t.pivot != nil {
			c.Check(lf.pivot, check.DeepEquals, t.pivot)
			c.Check(lf.sign, check.Equals, t.sign)
		}

		l := lf.L()
		if t.l != nil {
			c.Check(Equal(l, t.l), check.Equals, true)
		}
		u := lf.U()
		if t.u != nil {
			c.Check(Equal(u, t.u), check.Equals, true)
		}

		l = Mult(l, u, nil)
		c.Check(Approx(l, pivotRows(Clone(t.a), lf.pivot), 1e-12), check.Equals, true)

		aInv := Inv(Clone(t.a), nil)
		aInv = Mult(aInv, t.a, nil)
		c.Check(Approx(aInv, eye(3), 1e-12), check.Equals, true)

		x := lf.Solve(eye(3))
		t.a = Mult(t.a, x, nil)
		c.Check(Approx(t.a, eye(3), 1e-12), check.Equals, true)
	}
}
