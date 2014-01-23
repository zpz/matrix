// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dense

import (
	check "launchpad.net/gocheck"
)

func (s *S) TestCholesky(c *check.C) {
	for _, t := range []struct {
		a   *Dense
		spd bool
	}{
		{
			a: make_dense(3, 3, []float64{
				4, 1, 1,
				1, 2, 3,
				1, 3, 6,
			}),

			spd: true,
		},
	} {
		cl, ok := Chol(t.a)
		c.Check(ok, check.Equals, t.spd)

		lc := Mult(cl.L, T(cl.L, nil), nil)
		c.Check(EqualApprox(lc, t.a, 1e-12), check.Equals, true)

		ta := Mult(t.a, cl.Solve(eye(3)), nil)
		c.Check(EqualApprox(ta, eye(3), 1e-12), check.Equals, true)

		cr, ok := CholR(t.a)
		c.Check(ok, check.Equals, t.spd)

		rc := Mult(T(cr.U, nil), cr.U, nil)
		c.Check(EqualApprox(rc, t.a, 1e-12), check.Equals, true)

		tb := Mult(cr.Solve(eye(3)), t.a, nil)
		c.Check(EqualApprox(tb, eye(3), 1e-12), check.Equals, true)
	}
}
