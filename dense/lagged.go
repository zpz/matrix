package dense

type LaggedFloat64Slice struct {
	data   []float64
	stride int
}

func NewLaggedFloat64Slice(data []float64, stride int) *LaggedFloat64Slice {
	if data == nil || len(data) < 1 {
		panic(errInLength)
	}
	if stride < 1 {
		panic(err("stride must be positive"))
	}

	n := (len(data) - 1) / stride
	return &LaggedFloat64Slice{
		data[:1+n*stride],
		stride}
}

func (me *LaggedFloat64Slice) Len() int {
	return (len(me.data)-1)/me.stride + 1
}

func (me *LaggedFloat64Slice) Get(i int) float64 {
	// TODO: skipping bound check.
	// Is bound checking expensive?
	return me.data[i*me.stride]
}

func (me *LaggedFloat64Slice) Set(i int, val float64) *LaggedFloat64Slice {
	// TODO: skipping bound check.
	// Is bound checking expensive?
	me.data[i*me.stride] = val
	return me
}

func (me *LaggedFloat64Slice) Less(i, j int) bool {
	return me.Get(i) < me.Get(j)
}

func (me *LaggedFloat64Slice) Swap(i, j int) *LaggedFloat64Slice {
	x, y := me.data[i*me.stride], me.data[j*me.stride]
	me.data[i*me.stride], me.data[j*me.stride] = y, x
	return me
}

func (me *LaggedFloat64Slice) CopyFrom(in []float64) *LaggedFloat64Slice {
	n := me.Len()
	if len(in) != n {
		panic(errInLength)
	}
	for i, j := 0, 0; i < n; i, j = i+1, j+me.stride {
		me.data[j] = in[i]
	}
	return me
}

func (me *LaggedFloat64Slice) CopyTo(out []float64) []float64 {
	n := me.Len()
	out = use_slice(out, n, errOutLength)
	for i, j := 0, 0; i < n; i, j = i+1, j+me.stride {
		out[i] = me.data[j]
	}
	return out
}

func (me *LaggedFloat64Slice) Fill(val float64) *LaggedFloat64Slice {
	n := len(me.data)
	for i := 0; i < n; i += me.stride {
		me.data[i] = val
	}
	return me
}

// Sub returns a view that includes the elements [from, to).
func (me *LaggedFloat64Slice) Sub(from, to int) *LaggedFloat64Slice {
	n := me.Len()
	if from >= n || to > n || from >= to {
		panic(err("wrong input values"))
	}
	from = from * me.stride
	to = (to-1)*me.stride + 1
	return &LaggedFloat64Slice{
		me.data[from:to],
		me.stride}
}
