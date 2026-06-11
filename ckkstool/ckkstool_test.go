package ckkstool

import (
	"fmt"
	"lstm/utils"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// go test -v -run ^TestMatrixMultiply$ -timeout=20m lstm/ckkstool
func TestMatrixMultiply(t *testing.T) {
	ckksTool, err := NewCKKSToolForConfig(512, 64)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}
	// ckksTool.GenerateBatchGaloisKeys(512, 64)

	m, n, p := 512, 64, 64
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

	fmt.Println("数据生成成功")

	xcol := utils.ColCoding(x)
	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(xcol, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)

	var all time.Duration

	for i := 0; i < 8; i++ {
		start := time.Now()
		res := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(m, ctx, w, 8)
		// res := ckksTool.MatrixMultiplyWithWorkers(m, ctx, w, 16)
		elapsed := time.Since(start)
		fmt.Println("矩阵乘法时间:", elapsed)
		all += elapsed
		// ckksTool.LogCiphertextInfo(res, "res", 0)
		ckksTool.DecToFloat64(res)
		// res = ckksTool.MatrixMultiplyWithWorkers(m, ctx, w, 16)
		// ckksTool.DecToFloat64(res)
	}
	fmt.Println("all time:", all)
	// start := time.Now()
	// res := ckksTool.MatrixMultiplyCPMMDiagonalBSGS(m, ctx, w, 1)
	// // res := ckksTool.MatrixMultiplyWithWorkers(m, ctx, w, 16)
	// elapsed := time.Since(start)
	// fmt.Println("矩阵乘法时间:", elapsed)
	// ckksTool.LogCiphertextInfo(res, "res", 0)
	// ckksTool.DecToFloat64(res)

	y := make([]float64, 5)
	for i := 0; i < 5; i++ {
		y[i] = 0
		for j := 0; j < n; j++ {
			y[i] += x[i][j] * w[j][0]
		}
	}
	fmt.Println("pt:", y)
}

// go test -v -run ^TestMatrixMultiply1$ -timeout=20m lstm/ckkstool
func TestMatrixMultiply1(t *testing.T) {
	m, n, p := 512, 64, 64
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
	wt := utils.Transpose(w)
	fmt.Println("数据生成成功")

	ckksTool, err := NewCKKSToolForConfig(512, 64)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}

	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(utils.ColCoding(x), ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)
	fmt.Println(ctx.Level(), " ", ctx.Scale)

	ptx1 := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(utils.ColCoding(utils.PadMatrix(x, m, n)), ptx1)
	ctx1, _ := ckksTool.Enc.EncryptNew(ptx1)

	start := time.Now()
	res := ckksTool.MatrixMultiplyPCMMDiagonalBSGS_old(m, ctx1, utils.PadMatrix(wt, n, n), 2)
	elapsed := time.Since(start)
	fmt.Println("矩阵乘法1时间:", elapsed)

	ckksTool.DecToFloat64(res)
	ckksTool.Eval.RescaleTo(res, ctx.Scale, res)
	fmt.Println(res.Level(), " ", res.Scale)

	start = time.Now()
	res = ckksTool.MatrixMultiplyPCMMDiagonalBSGS(m, ctx1, utils.PadMatrix(wt, n, n), 2)
	elapsed = time.Since(start)
	fmt.Println("矩阵乘法2时间:", elapsed)

	ckksTool.DecToFloat64(res)
	ckksTool.Eval.RescaleTo(res, ctx.Scale, res)
	fmt.Println(res.Level(), " ", res.Scale)

	start = time.Now()
	res = ckksTool.MatrixMultiplyWithWorkers(m, ctx1, utils.PadMatrix(wt, n, n), 2)
	elapsed = time.Since(start)
	fmt.Println("矩阵乘法3时间:", elapsed)

	ckksTool.DecToFloat64(res)
	ckksTool.Eval.RescaleTo(res, ctx.Scale, res)
	fmt.Println(res.Level(), " ", res.Scale)

	y := make([]float64, 10)
	for i := 0; i < 5; i++ {
		y[i*2] = 0
		for j := 0; j < p; j++ {
			y[i*2] += x[i][j] * w[0][j]
		}
		for j := 0; j < p; j++ {
			y[i*2+1] += x[m-5+i][j] * w[n-1][j]
		}
	}
	fmt.Println(y)
}

// go test -v -run ^TestBootstrapping$ -timeout=20m lstm/ckkstool
func TestBootstrapping(t *testing.T) {
	ckksTool, err := NewCKKSToolForConfig(512, 64)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}

	valuesWant := make([]float64, ckksTool.Params.MaxSlots())
	for i := range valuesWant {
		valuesWant[i] = rand.Float64()
	}

	ptx := ckks.NewPlaintext(ckksTool.Params, 0)
	ckksTool.End.Encode(valuesWant, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)

	ckksTool.DecToFloat64(ctx)
	fmt.Println(ctx.Level(), ctx.Scale)

	start := time.Now()
	res, err := ckksTool.BootEval.Bootstrap(ctx)
	if err != nil {
		print(err)
	}
	elapsed := time.Since(start)
	fmt.Println("bootstrapping时间:", elapsed)
	ckksTool.DecToFloat64(res)
	fmt.Println(res.Level(), res.Scale)
}
