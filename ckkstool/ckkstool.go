package ckkstool

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
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

// func NewCKKSTool() (*CKKSTool, error) {
// 	return NewCKKSToolForConfig(512, 64)
// }

func RequiredRotationSteps(batchSize, hiddenDim int) []int {
	if batchSize <= 0 || hiddenDim <= 0 {
		return nil
	}

	steps := map[int]bool{}
	add := func(step int) {
		if step != 0 {
			steps[step] = true
		}
	}

	n2 := int(math.Ceil(math.Sqrt(float64(hiddenDim))))
	n1 := int(math.Ceil(float64(hiddenDim) / float64(n2)))
	for j := 1; j < n2; j++ {
		add(j * batchSize)
	}
	for i := 1; i < n1; i++ {
		add(i * n2 * batchSize)
	}

	for step := batchSize * (hiddenDim / 2); step >= batchSize && step > 0; step /= 2 {
		add(step)
	}
	for step := batchSize; step <= batchSize*(hiddenDim/2) && step > 0; step *= 2 {
		add(-step)
	}

	res := make([]int, 0, len(steps))
	for step := range steps {
		res = append(res, step)
	}
	sort.Ints(res)
	return res
}

func NewCKKSToolForConfig(batchSize, hiddenDim int) (*CKKSTool, error) {
	// Lattigo v6 参数设置
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 15,
		LogQ: []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		// LogQ:            []int{55, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		RingType:        ring.ConjugateInvariant,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		return nil, fmt.Errorf("param create error: %v", err)
	}

	bootParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(16),
		LogP:     []int{61, 61, 61, 61},
		Xs:       params.Xs(),
		LogSlots: utils.Pointy(15),
	})
	if err != nil {
		return nil, fmt.Errorf("bootstraping param create error: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	galEls := make([]uint64, 0)
	// for i := 1; i < 128; i++ {
	// 	galEls = append(galEls, params.GaloisElement(i*256))
	// }

	// for i := 1; i < 64; i++ {
	// 	galEls = append(galEls, params.GaloisElement(i*512))
	// }

	steps := RequiredRotationSteps(batchSize, hiddenDim)

	galElsSet := make(map[uint64]bool)
	for _, step := range steps {
		galElsSet[params.GaloisElement(step)] = true
	}

	for el := range galElsSet {
		galEls = append(galEls, el)
	}

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

func (ckksTool *CKKSTool) MatrixMultiplyPCMMDiagonalBSGS(m int, ctx *rlwe.Ciphertext, w [][]float64, numWorkers int) *rlwe.Ciphertext {
	n := len(w)
	slots := ckksTool.Params.MaxSlots()

	n2 := int(math.Ceil(math.Sqrt(float64(n))))    // Baby steps
	n1 := int(math.Ceil(float64(n) / float64(n2))) // Giant steps

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	X_baby := make([]*rlwe.Ciphertext, n2)
	X_baby[0] = ctx
	if n2 > 1 {
		rotations := make([]int, n2-1)
		for j := 1; j < n2; j++ {
			rotations[j-1] = j * m
		}
		rotated, err := ckksTool.Eval.RotateHoistedNew(ctx, rotations)
		if err != nil {
			log.Fatal(err)
		}
		for j := 1; j < n2; j++ {
			X_baby[j] = rotated[j*m]
		}
	}

	if numWorkers > n1 {
		numWorkers = n1
	}

	jobs := make(chan int)
	results := make(chan *rlwe.Ciphertext, numWorkers)
	var wg sync.WaitGroup

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localEval := ckks.NewEvaluator(ckksTool.Params, ckksTool.Evk)
			localEncoder := ckks.NewEncoder(ckksTool.Params)
			shiftedVector := make([]float64, slots)
			diagonalPt := ckks.NewPlaintext(ckksTool.Params, ctx.Level())

			encodeDiagonal := func(k int) *rlwe.Plaintext {
				giant := k / n2
				shift := (-giant * n2 * m) % slots
				if shift < 0 {
					shift += slots
				}

				for idx := range shiftedVector {
					shiftedVector[idx] = 0
				}
				for jMat := 0; jMat < n; jMat++ {
					rowW := (jMat + k) % n
					base := jMat * m
					value := w[rowW][jMat]
					for rowX := 0; rowX < m; rowX++ {
						slotIdx := base + rowX
						dstIdx := slotIdx - shift
						if dstIdx < 0 {
							dstIdx += slots
						}
						shiftedVector[dstIdx] = value
					}
				}

				localEncoder.Encode(shiftedVector, diagonalPt)
				return diagonalPt
			}

			for i := range jobs {
				var innerSum *rlwe.Ciphertext
				first := true

				// sum( X_baby[j] * W_shifted[i*n2 + j] )
				for j := 0; j < n2; j++ {
					k := i*n2 + j
					if k >= n {
						break
					}
					wDiagonal := encodeDiagonal(k)

					if first {
						innerSum = ckks.NewCiphertext(ckksTool.Params, X_baby[j].Degree(), X_baby[j].Level())
						innerSum.Scale = X_baby[j].Scale.Mul(wDiagonal.Scale)
						first = false
					}
					if err := localEval.MulThenAdd(X_baby[j], wDiagonal, innerSum); err != nil {
						log.Fatal(err)
					}
				}

				if innerSum != nil {
					localEval.Rescale(innerSum, innerSum)
				}

				rotAmount := i * n2 * m
				if rotAmount > 0 && innerSum != nil {
					if err := localEval.Rotate(innerSum, rotAmount, innerSum); err != nil {
						log.Fatalf("bsgs giant rotation failed for step %d: %v", rotAmount, err)
					}
				}

				results <- innerSum
			}
		}()
	}

	go func() {
		for i := 0; i < n1; i++ {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	var finalCtg *rlwe.Ciphertext
	firstAdd := true
	for result := range results {
		if result != nil {
			if firstAdd {
				finalCtg = result
				firstAdd = false
			} else {
				err := ckksTool.Eval.Add(finalCtg, result, finalCtg)
				if err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	return finalCtg
}

// MatrixMultiplyPCMMDiagonalBSGS 使用 BSGS 优化密文 * 明文对角线矩阵乘法，密文*明文，x[m, n], w[n, n]
func (ckksTool *CKKSTool) MatrixMultiplyPCMMDiagonalBSGS_old(m int, ctx *rlwe.Ciphertext, w [][]float64, numWorkers int) *rlwe.Ciphertext {
	n := len(w)
	slots := ckksTool.Params.MaxSlots()

	n2 := int(math.Ceil(math.Sqrt(float64(n))))
	n1 := int(math.Ceil(float64(n) / float64(n2)))

	WDiagonalsShifted := make([]*rlwe.Plaintext, n)
	for k := 0; k < n; k++ {
		i := k / n2 // Giant step 索引

		diagVector := make([]float64, slots)
		for rowX := 0; rowX < m; rowX++ {
			for j_mat := 0; j_mat < n; j_mat++ {
				rowW := (j_mat + k) % n
				slotIdx := (j_mat * m) + rowX
				diagVector[slotIdx] = w[rowW][j_mat]
			}
		}

		shift := (-i * n2 * m) % slots
		if shift < 0 {
			shift += slots
		}

		shiftedVector := make([]float64, slots)
		for idx := 0; idx < slots; idx++ {
			shiftedVector[idx] = diagVector[(idx+shift)%slots]
		}

		pt := ckks.NewPlaintext(ckksTool.Params, ctx.Level())
		ckksTool.End.Encode(shiftedVector, pt)
		WDiagonalsShifted[k] = pt
	}

	X_baby := make([]*rlwe.Ciphertext, n2)
	X_baby[0] = ctx.CopyNew()
	for j := 1; j < n2; j++ {
		var err error
		X_baby[j], err = ckksTool.Eval.RotateNew(ctx, j*m)
		if err != nil {
			log.Fatalf("old bsgs baby rotation failed for step %d: %v", j*m, err)
		}
	}

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	if numWorkers > n1 {
		numWorkers = n1
	}

	giantStepResults := make([]*rlwe.Ciphertext, n1)
	jobs := make(chan int, n1)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localEval := ckks.NewEvaluator(ckksTool.Params, ckksTool.Evk)

			for i := range jobs {
				var innerSum *rlwe.Ciphertext
				first := true

				for j := 0; j < n2; j++ {
					k := i*n2 + j
					if k >= n {
						break
					}

					tmp, _ := localEval.MulNew(X_baby[j], WDiagonalsShifted[k])
					localEval.Rescale(tmp, tmp)

					if first {
						innerSum = tmp
						first = false
					} else {
						localEval.Add(innerSum, tmp, innerSum)
					}
				}

				rotAmount := i * n2 * m
				if rotAmount > 0 && innerSum != nil {
					if err := localEval.Rotate(innerSum, rotAmount, innerSum); err != nil {
						log.Fatalf("old bsgs giant rotation failed for step %d: %v", rotAmount, err)
					}
				}

				giantStepResults[i] = innerSum
			}
		}()
	}

	for i := 0; i < n1; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	var finalCtg *rlwe.Ciphertext
	firstAdd := true
	for i := 0; i < n1; i++ {
		if giantStepResults[i] != nil {
			if firstAdd {
				finalCtg = giantStepResults[i].CopyNew()
				firstAdd = false
			} else {
				err := ckksTool.Eval.Add(finalCtg, giantStepResults[i], finalCtg)
				if err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	return finalCtg
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
			w1 := make([]float64, slots)
			ptW := ckks.NewPlaintext(ckksTool.Params, ctx.Level())
			var temp *rlwe.Ciphertext

			for k := range jobs {
				clear(w1)
				for i := 0; i < p; i++ {
					for j := 0; j < m; j++ {
						w1[i*m+j] = w[k][i]
					}
				}

				localEnd.Encode(w1, ptW)

				ctM, _ := localEval.MulNew(ctx, ptW)
				localEval.Rescale(ctM, ctM)

				rot := m
				for rot < m*p {
					if temp == nil {
						temp = ctM.CopyNew()
					} else {
						// Lattigo Copy is dst.Copy(src).
						temp.Copy(ctM)
					}
					if err := localEval.Rotate(temp, rot, temp); err != nil {
						log.Fatalf("fc reduction rotation failed for step %d: %v", rot, err)
					}
					localEval.Add(ctM, temp, ctM)
					rot = rot * 2
				}

				ctgs[k], _ = localEval.MulNew(ctM, ptM)
				localEval.Rescale(ctgs[k], ctgs[k])
				if err := localEval.Rotate(ctgs[k], -k*m, ctgs[k]); err != nil {
					log.Fatalf("fc output rotation failed for class %d, step %d: %v", k, -k*m, err)
				}
			}
		}()
	}

	for k := 0; k < n; k++ {
		jobs <- k
	}
	close(jobs)

	wg.Wait()

	ctg := ctgs[0]
	for i := 1; i < n; i++ {
		ckksTool.Eval.Add(ctg, ctgs[i], ctg)
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
	mask := make([]float64, slots)
	for i := 0; i < slots; i++ {
		if i < m {
			mask[i] = 1.0 / float64(n)
		}
	}
	return ckksTool.MeanWithMask(m, n, ctx, mask)
}

func (ckksTool *CKKSTool) MeanWithMask(m int, n int, ctx *rlwe.Ciphertext, mask []float64) *rlwe.Ciphertext {
	res := ctx.CopyNew()
	return ckksTool.MeanWithMaskInPlace(m, n, res, mask)
}

func (ckksTool *CKKSTool) MeanWithMaskInPlace(m int, n int, res *rlwe.Ciphertext, mask []float64) *rlwe.Ciphertext {
	temp := res.CopyNew()
	for i := m * (n / 2); i >= m; i = i / 2 {
		temp.Copy(res)
		if err := ckksTool.Eval.Rotate(temp, i, temp); err != nil {
			log.Fatalf("mean positive rotation failed for step %d: %v", i, err)
		}
		ckksTool.Eval.Add(res, temp, res)
	}

	ptM := ckks.NewPlaintext(ckksTool.Params, res.Level())
	ckksTool.End.Encode(mask, ptM)
	ckksTool.Eval.Mul(res, ptM, res)
	ckksTool.Eval.Rescale(res, res)

	for i := m; i <= m*(n/2); i = i * 2 {
		temp.Copy(res)
		if err := ckksTool.Eval.Rotate(temp, -i, temp); err != nil {
			log.Fatalf("mean negative rotation failed for step %d: %v", -i, err)
		}
		ckksTool.Eval.Add(res, temp, res)
	}

	return res
}

// 求均方值 (Mean Square)
func (ckksTool *CKKSTool) MeanSquare(m int, n int, ctx *rlwe.Ciphertext) *rlwe.Ciphertext {
	slots := ckksTool.Params.MaxSlots()
	mask := make([]float64, slots)
	for i := 0; i < slots; i++ {
		if i < m {
			mask[i] = 1.0 / float64(n)
		}
	}
	return ckksTool.MeanSquareWithMask(m, n, ctx, mask)
}

func (ckksTool *CKKSTool) MeanSquareWithMask(m int, n int, ctx *rlwe.Ciphertext, mask []float64) *rlwe.Ciphertext {
	squaredCtx, err := ckksTool.Eval.MulRelinNew(ctx, ctx)
	ckksTool.Eval.Rescale(squaredCtx, squaredCtx)
	if err != nil {
		fmt.Println("MeanSquare error:", err)
		return nil
	}
	return ckksTool.MeanWithMaskInPlace(m, n, squaredCtx, mask)
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
	for i := 0; i < 10; i++ {
		fmt.Print(res[i], " ")
	}
	fmt.Println()
}

func (ckksTool *CKKSTool) LogCiphertextInfo(ct *rlwe.Ciphertext, name string, t int) {
	if ct == nil {
		fmt.Printf("[seqlen%d] %s: ct is null\n", t, name)
		return
	}

	pt := ckksTool.Dec.DecryptNew(ct)
	res := make([]float64, ckksTool.Params.MaxSlots())
	ckksTool.End.Decode(pt, res)

	var sum, max, min float64
	var nanCount, infCount int
	min = res[0]
	max = res[0]

	for i := 0; i < len(res); i++ {
		val := res[i]
		if math.IsNaN(val) {
			nanCount++
		} else if math.IsInf(val, 0) {
			infCount++
		} else {
			sum += val
			if val > max {
				max = val
			}
			if val < min {
				min = val
			}
		}
	}

	fmt.Printf("[seqlen %d] %s - static: Sum=%.6f, Max=%.6f, Min=%.6f, NaN=%d, Inf=%d\n",
		t, name, sum, max, min, nanCount, infCount)

	fmt.Printf("[seqlen %d] %s - five: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
		t, name, res[0], res[1], res[2], res[3], res[4], res[32763], res[32764], res[32765], res[32766], res[32767])
}
