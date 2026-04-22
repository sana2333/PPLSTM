package pcmm

import (
	"fmt"
	"lstm/utils"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// go test -v -run ^TestMatrixMultiply$ -timeout=20m lstm/pcmm
func TestMatrixMultiply(t *testing.T) {
	ckksTool, err := NewCKKSTool()
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("init sucess")
	}

	// m, n, p := 512, 64, 64
	m, n, p := 256, 128, 128
	x := make([][]float64, m)
	w := make([][]float64, n)
	for i := 0; i < m; i++ {
		x[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			x[i][j] = rand.Float64()
		}
	}
	for i := 0; i < n; i++ {
		w[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			w[i][j] = rand.Float64()
		}
	}

	xcol := utils.ColCoding(x)
	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(xcol, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)

	var all time.Duration

	for i := 0; i < 10; i++ {
		start := time.Now()
		ckksTool.MatrixMultiplyPCMMDiagonalBSGS(m, ctx, w, 1)
		elapsed := time.Since(start)
		fmt.Println(i, " pcmm time:", elapsed)
		all += elapsed
	}
	fmt.Println("all time:", all)
	fmt.Println("pcmm time:", all/10)
}
