package ckkstool

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type CKKSTool struct {
	Params   ckks.Parameters
	End      *ckks.Encoder
	Enc      *rlwe.Encryptor
	Eval     *ckks.Evaluator
	Dec      *rlwe.Decryptor
	Evk      *rlwe.MemEvaluationKeySet
	BootEval *bootstrapping.Evaluator
}

func NewCKKSTool() (*CKKSTool, error) {
	// Lattigo v6 参数设置
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		RingType:        ring.ConjugateInvariant,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		return nil, fmt.Errorf("param creat error: %v", err)
	}

	bootParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(16),
		LogP:     []int{61, 61, 61, 61},
		Xs:       params.Xs(),
		LogSlots: utils.Pointy(15),
	})
	if err != nil {
		return nil, fmt.Errorf("bootstraping param creat error: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	galEls := make([]uint64, 0)
	for i := 256; i < 256*128; i = i * 2 {
		galEls = append(galEls, params.GaloisElement(i))
	}
	for i := 1; i < 128; i++ {
		galEls = append(galEls, params.GaloisElement(i*256))
	}

	// for i := 512; i < 512*64; i = i * 2 {
	// 	galEls = append(galEls, params.GaloisElement(i))
	// }
	// for i := 1; i < 64; i++ {
	// 	galEls = append(galEls, params.GaloisElement(i*512))
	// }

	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk), kgen.GenGaloisKeysNew(galEls, sk)...)
	end := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	eval := ckks.NewEvaluator(params, evk)
	dec := rlwe.NewDecryptor(params, sk)

	bootEvk, _, _ := bootParams.GenEvaluationKeys(sk)
	bootEval, err := bootstrapping.NewEvaluator(bootParams, bootEvk)
	if err != nil {
		return nil, fmt.Errorf("init Bootstrapper error: %v", err)
	}

	return &CKKSTool{
		Params:   params,
		End:      end,
		Enc:      enc,
		Eval:     eval,
		Dec:      dec,
		Evk:      evk,
		BootEval: bootEval,
	}, nil
}

// 矩阵乘法多线程版, x[m, p], w[n, p]
func (ckksTool *CKKSTool) MatrixMultiplyWithWorkers(m int, ctx *rlwe.Ciphertext, w [][]float64, numWorkers int) *rlwe.Ciphertext {
	n := len(w)
	p := len(w[0])
	ctgs := make([]*rlwe.Ciphertext, n)
	slots := ckksTool.Params.MaxSlots()

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	if numWorkers > n {
		numWorkers = n
	}

	mask := make([]float64, slots)
	for i := 0; i < m; i++ {
		if i < m {
			mask[i] = 1
		} else {
			mask[i] = 0
		}
	}
	ptM := ckks.NewPlaintext(ckksTool.Params, ctx.Level()-1)
	ckksTool.End.Encode(mask, ptM)

	jobs := make(chan int, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			localEval := ckks.NewEvaluator(ckksTool.Params, ckksTool.Evk)
			localEnd := ckks.NewEncoder(ckksTool.Params)

			for k := range jobs {
				w1 := make([]float64, slots)
				for i := 0; i < p; i++ {
					for j := 0; j < m; j++ {
						w1[i*m+j] = w[k][i]
					}
				}

				ptW := ckks.NewPlaintext(ckksTool.Params, ctx.Level())
				localEnd.Encode(w1, ptW)

				ctM, _ := localEval.MulNew(ctx, ptW)
				localEval.Rescale(ctM, ctM)

				rot := m
				for rot < m*p {
					temp := ctM.CopyNew()
					localEval.Rotate(temp, rot, temp)
					localEval.Add(ctM, temp, ctM)
					rot = rot * 2
				}

				ctgs[k], _ = localEval.MulNew(ctM, ptM)
				localEval.Rescale(ctgs[k], ctgs[k])
				localEval.Rotate(ctgs[k], -k*m, ctgs[k])
			}
		}()
	}

	for k := 0; k < n; k++ {
		jobs <- k
	}
	close(jobs)

	wg.Wait()

	ctg := ctgs[0].CopyNew()
	for i := 1; i < n; i++ {
		ckksTool.Eval.Add(ctg, ctgs[i], ctg)
	}

	return ctg
}

// 对角线编码,x[m, n], w[n, n]
func (ckksTool *CKKSTool) MatrixMultiplyPCMMDiagonal(m int, ctx *rlwe.Ciphertext, w [][]float64, numWorkers int) *rlwe.Ciphertext {

	n := len(w)
	slots := ckksTool.Params.MaxSlots()

	// 明文矩阵 W 的对角线编码
	WDiagonals := make([]*rlwe.Plaintext, n)
	for k := 0; k < n; k++ {
		diagVector := make([]float64, slots)

		for rowX := 0; rowX < m; rowX++ {
			for j := 0; j < n; j++ {
				rowW := (j + k) % n // W 的行索引 rowW = (j + k) mod n
				slotIdx := (j * m) + rowX
				diagVector[slotIdx] = w[rowW][j]
			}
		}

		pt := ckks.NewPlaintext(ckksTool.Params, ctx.Level())
		ckksTool.End.Encode(diagVector, pt)
		WDiagonals[k] = pt
	}

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	if numWorkers > n {
		numWorkers = n
	}

	ctgs := make([]*rlwe.Ciphertext, n)
	jobs := make(chan int, n)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			localEval := ckks.NewEvaluator(ckksTool.Params, ckksTool.Evk)

			for k := range jobs {
				var cXRotated *rlwe.Ciphertext
				if k == 0 {
					cXRotated = ctx.CopyNew()
				} else {
					cXRotated, _ = localEval.RotateNew(ctx, k*m)
				}

				ctgs[k], _ = localEval.MulNew(cXRotated, WDiagonals[k])
				localEval.Rescale(ctgs[k], ctgs[k])
			}
		}()
	}

	for k := 0; k < n; k++ {
		jobs <- k
	}
	close(jobs)
	wg.Wait()

	ctg := ctgs[0].CopyNew()

	for i := 1; i < n; i++ {
		err := ckksTool.Eval.Add(ctg, ctgs[i], ctg)
		if err != nil {
			log.Fatal(err)
		}
	}

	return ctg
}

func (ckksTool *CKKSTool) OptimizedFit(op *rlwe.Ciphertext, coeffs []float64, interval [2]float64) *rlwe.Ciphertext {
	bignumPoly := bignum.NewPolynomial(bignum.Monomial, coeffs, interval)
	PolyEval := polynomial.NewEvaluator(ckksTool.Params, ckksTool.Eval)
	poly := polynomial.NewPolynomial(bignumPoly)
	targetScale := ckksTool.Params.DefaultScale()
	res, err := PolyEval.Evaluate(op, poly, targetScale)
	if err != nil {
		fmt.Println("poly error: ", err)
	}
	return res
}

func (ckksTool *CKKSTool) OneToPt(x float64, level int) *rlwe.Plaintext {
	X := make([]float64, ckksTool.Params.MaxSlots())
	for i := 0; i < len(X); i++ {
		X[i] = x
	}
	ptx := ckks.NewPlaintext(ckksTool.Params, level)
	ckksTool.End.Encode(X, ptx)
	return ptx
}

func (ckksTool *CKKSTool) ArrayToPt(x []float64, level int) *rlwe.Plaintext {
	ptx := ckks.NewPlaintext(ckksTool.Params, level)
	ckksTool.End.Encode(x, ptx)
	return ptx
}

// 求mean
func (ckksTool *CKKSTool) Mean(m int, n int, ctx *rlwe.Ciphertext) *rlwe.Ciphertext {
	slots := ckksTool.Params.MaxSlots()
	res := ctx.CopyNew()
	for i := m * (n / 2); i >= m; i = i / 2 {
		temp := res.CopyNew()
		ckksTool.Eval.Rotate(temp, i, temp)
		ckksTool.Eval.Add(res, temp, res)
	}

	mask := make([]float64, slots)
	for i := 0; i < slots; i++ {
		if i < m {
			mask[i] = 1.0 / float64(n)
		} else {
			mask[i] = 0
		}
	}
	ptM := ckks.NewPlaintext(ckksTool.Params, res.Level())
	ckksTool.End.Encode(mask, ptM)
	ckksTool.Eval.Mul(res, ptM, res)
	ckksTool.Eval.Rescale(res, res)

	for i := m; i <= m*(n/2); i = i * 2 {
		temp := res.CopyNew()
		ckksTool.Eval.Rotate(temp, -i, temp)
		ckksTool.Eval.Add(res, temp, res)
	}

	return res
}

// 求均方值 (Mean Square)
func (ckksTool *CKKSTool) MeanSquare(m int, n int, ctx *rlwe.Ciphertext) *rlwe.Ciphertext {
	squaredCtx, err := ckksTool.Eval.MulRelinNew(ctx, ctx)
	ckksTool.Eval.Rescale(squaredCtx, squaredCtx)
	if err != nil {
		fmt.Println("MeanSquare error:", err)
		return nil
	}
	return ckksTool.Mean(m, n, squaredCtx)
}

// 求var
func (ckksTool *CKKSTool) Var(m int, n int, ctx *rlwe.Ciphertext, ctm *rlwe.Ciphertext) *rlwe.Ciphertext {
	ckksTool.Eval.DropLevel(ctx, ctx.Level()-ctm.Level())
	temp, _ := ckksTool.Eval.SubNew(ctx, ctm)
	res, err := ckksTool.Eval.MulRelinNew(temp, temp)
	ckksTool.Eval.Rescale(res, res)
	if err != nil {
		fmt.Println(err)
	}
	return ckksTool.Mean(m, n, res)
}

func (ckksTool *CKKSTool) LoadData(filename string) *rlwe.Ciphertext {
	file, err := os.Open(filename)
	if err != nil {
		panic(fmt.Sprintf("failed to open file: %v", err))
	}
	defer file.Close()

	var res []float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if err != nil {
			panic(fmt.Sprintf("invalid float in file: %v", err))
		}
		res = append(res, val)
	}
	if err := scanner.Err(); err != nil {
		panic(fmt.Sprintf("error reading file: %v", err))
	}
	// fmt.Println(res)
	ptr := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(res, ptr)
	ctr, _ := ckksTool.Enc.EncryptNew(ptr)
	return ctr
}

func (ckksTool *CKKSTool) DecToFloat64(ctx *rlwe.Ciphertext) {
	res := make([]float64, ckksTool.Params.MaxSlots())
	ptx := ckksTool.Dec.DecryptNew(ctx)
	ckksTool.End.Decode(ptx, res)
	for i := 0; i < 5; i++ {
		fmt.Print(res[i], " ")
		fmt.Print(res[32763+i], " ")
	}
	fmt.Println()
}
