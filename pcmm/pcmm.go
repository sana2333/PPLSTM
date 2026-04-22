package pcmm

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
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
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{55, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		RingType:        ring.ConjugateInvariant,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		return nil, fmt.Errorf("param create error: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	galEls := make([]uint64, 0)

	for i := 0; i < 12; i++ {
		galEls = append(galEls, params.GaloisElement(i*256))
	}
	for i := 0; i < 11; i++ {
		galEls = append(galEls, params.GaloisElement(i*12*256))
	}

	for i := 0; i < 8; i++ {
		galEls = append(galEls, params.GaloisElement(i*512))
	}
	for i := 0; i < 8; i++ {
		galEls = append(galEls, params.GaloisElement(i*8*512))
	}

	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk), kgen.GenGaloisKeysNew(galEls, sk)...)
	end := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	eval := ckks.NewEvaluator(params, evk)
	dec := rlwe.NewDecryptor(params, sk)

	return &CKKSTool{
		Params: params,
		End:    end,
		Enc:    enc,
		Eval:   eval,
		Dec:    dec,
		Evk:    evk,
	}, nil
}

func (ckksTool *CKKSTool) MatrixMultiplyPCMMDiagonalBSGS(m int, ctx *rlwe.Ciphertext, w [][]float64, numWorkers int) *rlwe.Ciphertext {
	n := len(w)
	slots := ckksTool.Params.MaxSlots()

	n2 := int(math.Ceil(math.Sqrt(float64(n))))
	n1 := int(math.Ceil(float64(n) / float64(n2)))

	WDiagonalsShifted := make([]*rlwe.Plaintext, n)
	for k := 0; k < n; k++ {
		i := k / n2

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
		X_baby[j], _ = ckksTool.Eval.RotateNew(ctx, j*m)
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
					localEval.Rotate(innerSum, rotAmount, innerSum)
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
