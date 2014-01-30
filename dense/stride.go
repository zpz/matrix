package dense

type Float64Stride struct {
	data   []float64
	stride int
}

func NewFloat64Stride(data []float64, stride int) *Float64Stride {
	if data == nil || len(data) < 1 {
		panic(errInLength)
	}
	if stride < 1 {
		panic(err("stride must be positive"))
	}

	n := (len(data) - 1) / stride
	return &Float64Stride{
		data[:1+n*stride],
		stride}
}

func (me *Float64Stride) Len() int {
	return (len(me.data)-1)/me.stride + 1
}

func (me *Float64Stride) Get(i int) float64 {
	// TODO: skipping bound check.
	// Is bound checking expensive?
	return me.data[i*me.stride]
}

func (me *Float64Stride) Set(i int, val float64) *Float64Stride {
	// TODO: skipping bound check.
	// Is bound checking expensive?
	me.data[i*me.stride] = val
	return me
}

func (me *Float64Stride) Less(i, j int) bool {
	return me.Get(i) < me.Get(j)
}

func (me *Float64Stride) Swap(i, j int) *Float64Stride {
	x, y := me.data[i*me.stride], me.data[j*me.stride]
	me.data[i*me.stride], me.data[j*me.stride] = y, x
	return me
}

func (me *Float64Stride) CopyFrom(in []float64) *Float64Stride {
	n := me.Len()
	if len(in) != n {
		panic(errInLength)
	}
	for i, j := 0, 0; i < n; i, j = i+1, j+me.stride {
		me.data[j] = in[i]
	}
	return me
}

func (me *Float64Stride) CopyTo(out []float64) []float64 {
	n := me.Len()
	out = use_slice(out, n, errOutLength)
	for i, j := 0, 0; i < n; i, j = i+1, j+me.stride {
		out[i] = me.data[j]
	}
	return out
}

func (me *Float64Stride) Fill(val float64) *Float64Stride {
	n := len(me.data)
	for i := 0; i < n; i += me.stride {
		me.data[i] = val
	}
	return me
}

// Sub returns a view that includes the elements [from, to).
func (me *Float64Stride) Sub(from, to int) *Float64Stride {
	n := me.Len()
	if from >= n || to > n || from >= to {
		panic(err("wrong input values"))
	}
	from = from * me.stride
	to = (to-1)*me.stride + 1
	return &Float64Stride{
		me.data[from:to],
		me.stride}
}
