package ckkstool

import (
	"fmt"
	"lstm/utils"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// go test -v -run ^TestPCMMDiagonal$ -timeout=20m lstm/ckkstool
func TestPCMMDiagonal(t *testing.T) {

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

	ckksTool, err := NewCKKSTool()
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
	res := ckksTool.MatrixMultiplyPCMMDiagonal(m, ctx1, utils.PadMatrix(wt, n, n), 2)
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

// go test -v -run ^TestMatrixMultiply1$ -timeout=20m lstm/ckkstool
func TestMatrixMultiply(t *testing.T) {
	m := 256

	params := utils.GetParams("dbpedia", 1)
	fmt.Println("load params finish")
	x, _ := utils.LoadTxtFloatVector("D:\\Code\\Golang\\lstm\\result\\dbpedia_rms_h_30.txt")
	w1 := params.FCWeight
	fmt.Println(len(w1))
	w, b := params.FCWeight, utils.Repeat(m, params.FCBias)
	fmt.Println("load x finish")
	fmt.Println(params.FCBias)
	fmt.Println(len(w), len(w[0]), len(b))

	ckksTool, err := NewCKKSTool()
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}

	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(x, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)
	ckksTool.DecToFloat64(ctx)

	start := time.Now()
	res := ckksTool.MatrixMultiplyWithWorkers(m, ctx, w, 4)
	elapsed := time.Since(start)
	fmt.Println("矩阵乘法时间:", elapsed)
	ckksTool.DecToFloat64(res)
	ckksTool.Eval.Add(res, b, res)
	ckksTool.DecToFloat64(res)

	logRes := make([]float64, ckksTool.Params.MaxSlots())
	ckksTool.End.Decode(ckksTool.Dec.DecryptNew(res), logRes)
	utils.SaveDataToFile(logRes, fmt.Sprintf("%s_rms_%d.txt", "dbpedia", 30))
}

func TestMatrixMultiply1(t *testing.T) {
	ckksTool, err := NewCKKSTool()
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}

	m, n, p := 256, 4, 128
	// x := make([][]float64, m)
	w := make([][]float64, n)
	b := make([]float64, n)
	// for i := 0; i < m; i++ {
	// 	x[i] = make([]float64, p)
	// 	for j := 0; j < p; j++ {
	// 		x[i][j] = rand.Float64()
	// 	}
	// }
	for i := 0; i < n; i++ {
		b[i] = rand.Float64()
		w[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			w[i][j] = rand.Float64()
		}
	}

	x, _ := utils.LoadTxtFloatVector("D:\\Code\\Golang\\lstm\\result\\agnews_rms_h_30.txt")

	fmt.Println("数据生成成功")

	// xcol := utils.ColCoding(x)
	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	// ckksTool.End.Encode(xcol, ptx)
	ckksTool.End.Encode(x, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)
	fmt.Println(ctx.Level(), " ", ctx.Scale)

	start := time.Now()
	res := ckksTool.MatrixMultiplyWithWorkers(m, ctx, w, 4)
	elapsed := time.Since(start)
	fmt.Println("矩阵乘法时间:", elapsed)
	ckksTool.Eval.Add(res, utils.Repeat(m, b), res)

	ckksTool.DecToFloat64(res)

	ckksTool.Eval.RescaleTo(res, ctx.Scale, res)
	fmt.Println(res.Level(), " ", res.Scale)

	y := make([]float64, 10)
	for i := 0; i < 10; i++ {
		for j := 0; j < p; j++ {
			y[i] += x[i+j*m] * w[0][j]
		}
		y[i] += b[0]
	}
	fmt.Println(y)
}

// go test -v -run ^TestOptimizedFit$ -timeout=20m lstm/ckkstool
func TestOptimizedFit(t *testing.T) {
	rsqrt := []float64{2.50005249, -3.20120857, 2.01693233, -0.4280573}
	rsqrtRange := [2]float64{0.01, 2.4}

	ckksTool, err := NewCKKSTool()
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("初始化成功")
	}

	m, p := 512, 50
	x := make([][]float64, m)
	for i := 0; i < m; i++ {
		x[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			x[i][j] = rand.Float64()
		}
	}
	fmt.Println("数据生成成功")

	xcol := utils.ColCoding(x)
	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(xcol, ptx)
	ctx, _ := ckksTool.Enc.EncryptNew(ptx)

	ctx = ckksTool.OptimizedFit(ctx, rsqrt, rsqrtRange)
	ckksTool.DecToFloat64(ctx)
}

// go test -v -run ^TestBootstrapping$ -timeout=20m lstm/ckkstool
func TestBootstrapping(t *testing.T) {
	ckksTool, err := NewCKKSTool()
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
