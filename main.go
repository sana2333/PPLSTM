package main

import (
	"fmt"
	"lstm/ckkstool"
	"lstm/coeffs"
	"lstm/utils"

	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	MB  = 1024 * 1024
	GiB = 1024 * 1024 * 1024
)

type memoryModeConfig struct {
	enabled          bool
	gcPercent        int
	memoryLimitGiB   int64
	memoryLimitBytes int64
}

func configureMemoryMode() memoryModeConfig {
	cfg := memoryModeConfig{
		enabled:        true,
		gcPercent:      50,
		memoryLimitGiB: 30,
	}

	switch os.Getenv("PPLSTM_MEMORY_MODE") {
	case "0", "false", "False", "FALSE", "off", "Off", "OFF":
		cfg.enabled = false
	}

	if !cfg.enabled {
		fmt.Println("memory mode: disabled")
		return cfg
	}

	if value := os.Getenv("PPLSTM_GOGC"); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			cfg.gcPercent = parsed
		} else {
			fmt.Printf("invalid PPLSTM_GOGC=%q, using default %d\n", value, cfg.gcPercent)
		}
	}

	if value := os.Getenv("PPLSTM_GOMEMLIMIT_GIB"); value != "" {
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil && parsed > 0 {
			cfg.memoryLimitGiB = parsed
		} else {
			fmt.Printf("invalid PPLSTM_GOMEMLIMIT_GIB=%q, using default %d\n", value, cfg.memoryLimitGiB)
		}
	}

	cfg.memoryLimitBytes = cfg.memoryLimitGiB * GiB
	debug.SetGCPercent(cfg.gcPercent)
	debug.SetMemoryLimit(cfg.memoryLimitBytes)
	fmt.Printf("memory mode: enabled GOGC=%d GOMEMLIMIT=%d GiB\n", cfg.gcPercent, cfg.memoryLimitGiB)
	return cfg
}

func runMemoryBarrier(enabled bool, label string, trace bool) {
	if !enabled {
		return
	}
	if trace {
		logMemoryStats(label + " before memory barrier")
	}
	runtime.GC()
	if trace {
		logMemoryStats(label + " after memory barrier")
	}
}

func logMemoryStats(label string) runtime.MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("%s: Alloc=%.2f MB HeapInuse=%.2f MB HeapIdle=%.2f MB HeapReleased=%.2f MB Sys=%.2f MB NumGC=%d\n",
		label,
		float64(m.Alloc)/MB,
		float64(m.HeapInuse)/MB,
		float64(m.HeapIdle)/MB,
		float64(m.HeapReleased)/MB,
		float64(m.Sys)/MB,
		m.NumGC)
	return m
}

func lstm(dataName string, batchID int, hidden_dim int, thread int) {
	var start time.Time
	var elapsed time.Duration

	var maxAlloc uint64
	debugPrint := false
	memoryMode := configureMemoryMode()

	embeddingInput := utils.GetEmbeddingInput(dataName, batchID)
	batchSize := embeddingInput.BatchSize()
	seqLen := embeddingInput.SeqLen()

	// Generate batch-specific Galois keys (only the rotations needed for this batchSize and hiddenDim)
	fmt.Println(embeddingInput.EmbeddingDim(), " ", seqLen, " ", batchSize)
	seqLens := utils.GetSeqLen(dataName, batchID)
	minLen := slices.Min(seqLens)
	maxLen := slices.Max(seqLens)
	fmt.Println(maxLen)

	layers := 1
	params := utils.GetParams(dataName, layers)
	numClasses := len(params.FCBias)

	fmt.Println("ckks initting")
	fmt.Printf("rotation steps: %d\n", len(ckkstool.RequiredRotationSteps(batchSize, hidden_dim)))
	start = time.Now()
	ckksTool, err := ckkstool.NewCKKSToolForConfig(batchSize, hidden_dim)
	if err != nil {
		fmt.Println(err)
		return
	} else {
		elapsed = time.Since(start)
		fmt.Println("ckks init success and the running time: ", elapsed)
	}
	logMemoryStats("after ckks init")

	slots := ckksTool.Params.MaxSlots()

	coeff := coeffs.DatasetCoeffs[dataName]

	z := make([]float64, slots)
	ptz := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	ckksTool.End.Encode(z, ptz)

	cth := make([]*rlwe.Ciphertext, layers)
	for i := 0; i < layers; i++ {
		cth[i], _ = ckksTool.Enc.EncryptNew(ptz)
	}

	ctc := make([]*rlwe.Ciphertext, layers)
	for i := 0; i < layers; i++ {
		ctc[i], _ = ckksTool.Enc.EncryptNew(ptz)
	}

	var finalH *rlwe.Ciphertext

	w_ih_0 := make([][][]float64, layers)
	w_ih_1 := make([][][]float64, layers)
	w_ih_2 := make([][][]float64, layers)
	w_ih_3 := make([][][]float64, layers)
	b_ih_0 := make([][]float64, layers)
	b_ih_1 := make([][]float64, layers)
	b_ih_2 := make([][]float64, layers)
	b_ih_3 := make([][]float64, layers)

	w_hh_0 := make([][][]float64, layers)
	w_hh_1 := make([][][]float64, layers)
	w_hh_2 := make([][][]float64, layers)
	w_hh_3 := make([][][]float64, layers)
	b_hh_0 := make([][]float64, layers)
	b_hh_1 := make([][]float64, layers)
	b_hh_2 := make([][]float64, layers)
	b_hh_3 := make([][]float64, layers)

	w_rn := make([][]float64, layers)
	for i := 0; i < layers; i++ {
		w_ih := params.LSTM[i].W_ih
		b_ih := params.LSTM[i].B_ih
		w_ih_0[i], w_ih_1[i], w_ih_2[i], w_ih_3[i] = w_ih[0:hidden_dim], w_ih[hidden_dim:hidden_dim*2], w_ih[hidden_dim*2:hidden_dim*3], w_ih[hidden_dim*3:hidden_dim*4]

		w_ih_0[i] = utils.PadMatrix(utils.Transpose(w_ih_0[i]), hidden_dim, hidden_dim)
		w_ih_1[i] = utils.PadMatrix(utils.Transpose(w_ih_1[i]), hidden_dim, hidden_dim)
		w_ih_2[i] = utils.PadMatrix(utils.Transpose(w_ih_2[i]), hidden_dim, hidden_dim)
		w_ih_3[i] = utils.PadMatrix(utils.Transpose(w_ih_3[i]), hidden_dim, hidden_dim)

		b_ih_0[i] = utils.Repeat(batchSize, b_ih[0:hidden_dim])
		b_ih_1[i] = utils.Repeat(batchSize, b_ih[hidden_dim:hidden_dim*2])
		b_ih_2[i] = utils.Repeat(batchSize, b_ih[hidden_dim*2:hidden_dim*3])
		b_ih_3[i] = utils.Repeat(batchSize, b_ih[hidden_dim*3:hidden_dim*4])

		w_hh := params.LSTM[i].W_hh
		b_hh := params.LSTM[i].B_hh

		w_hh_0[i], w_hh_1[i], w_hh_2[i], w_hh_3[i] = w_hh[0:hidden_dim], w_hh[hidden_dim:hidden_dim*2], w_hh[hidden_dim*2:hidden_dim*3], w_hh[hidden_dim*3:hidden_dim*4]

		w_hh_0[i] = utils.Transpose(w_hh_0[i])
		w_hh_1[i] = utils.Transpose(w_hh_1[i])
		w_hh_2[i] = utils.Transpose(w_hh_2[i])
		w_hh_3[i] = utils.Transpose(w_hh_3[i])

		b_hh_0[i] = utils.Repeat(batchSize, b_hh[0:hidden_dim])
		b_hh_1[i] = utils.Repeat(batchSize, b_hh[hidden_dim:hidden_dim*2])
		b_hh_2[i] = utils.Repeat(batchSize, b_hh[hidden_dim*2:hidden_dim*3])
		b_hh_3[i] = utils.Repeat(batchSize, b_hh[hidden_dim*3:hidden_dim*4])

		w_rn[i] = utils.Repeat(batchSize, params.RMSNorm[i].Weight)
	}

	fmt.Println(len(params.FCWeight))
	fmt.Println(len(params.FCWeight[0]))

	xSlots := make([]float64, slots)
	ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
	finalMask := make([]float64, slots)
	meanMask := make([]float64, slots)
	for i := 0; i < batchSize; i++ {
		meanMask[i] = 1.0 / float64(hidden_dim)
	}

	startTime := time.Now()
	for t := 0; t < maxLen; t++ {
		fmt.Printf("\n start seqlen %d ===\n", t)
		start = time.Now()
		embeddingInput.FillTimeStepSlots(t, hidden_dim, xSlots)
		ckksTool.End.Encode(xSlots, ptx)
		ctx, _ := ckksTool.Enc.EncryptNew(ptx)

		for i := 0; i < layers; i++ {
			ctf_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_1[i], thread)
			ckksTool.Eval.Add(ctf_ih, ckksTool.ArrayToPt(b_ih_1[i], ctf_ih.Level()), ctf_ih)
			ctf_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_1[i], thread)
			ckksTool.Eval.Add(ctf_hh, ckksTool.ArrayToPt(b_hh_1[i], ctf_hh.Level()), ctf_hh)
			ctf, _ := ckksTool.Eval.AddNew(ctf_ih, ctf_hh)
			ctf_ih, ctf_hh = nil, nil

			F := ckksTool.OptimizedFit(ctf, coeff.Sigmoid, coeff.SigmoidRange)
			ctf = nil
			ckksTool.Eval.MulRelin(ctc[i], F, ctc[i])
			ckksTool.Eval.Rescale(ctc[i], ctc[i])
			F = nil

			cti_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_0[i], thread)
			ckksTool.Eval.Add(cti_ih, ckksTool.ArrayToPt(b_ih_0[i], cti_ih.Level()), cti_ih)
			cti_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_0[i], thread)
			ckksTool.Eval.Add(cti_hh, ckksTool.ArrayToPt(b_hh_0[i], cti_hh.Level()), cti_hh)
			cti, _ := ckksTool.Eval.AddNew(cti_ih, cti_hh)
			cti_ih, cti_hh = nil, nil

			I := ckksTool.OptimizedFit(cti, coeff.Sigmoid, coeff.SigmoidRange)
			cti = nil

			ctg_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_2[i], thread)
			ckksTool.Eval.Add(ctg_ih, ckksTool.ArrayToPt(b_ih_2[i], ctg_ih.Level()), ctg_ih)
			ctg_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_2[i], thread)
			ckksTool.Eval.Add(ctg_hh, ckksTool.ArrayToPt(b_hh_2[i], ctg_hh.Level()), ctg_hh)
			ctg, _ := ckksTool.Eval.AddNew(ctg_ih, ctg_hh)
			ctg_ih, ctg_hh = nil, nil

			G := ckksTool.OptimizedFit(ctg, coeff.TanhG, coeff.TanhGRange)
			ctg = nil

			temp, err := ckksTool.Eval.MulRelinNew(I, G)
			if err != nil {
				fmt.Println(err)
			}
			ckksTool.Eval.Rescale(temp, temp)
			ckksTool.Eval.Add(ctc[i], temp, ctc[i])
			I, G, temp = nil, nil, nil

			ms := ckksTool.MeanSquareWithMask(batchSize, hidden_dim, ctc[i], meanMask)
			ckksTool.Eval.Add(ms, 1e-5, ms)

			rv := ckksTool.OptimizedFit(ms, coeff.Rsqrt, coeff.RsqrtRange)
			ms = nil
			ckksTool.Eval.MulRelin(ctc[i], rv, ctc[i])
			rv = nil

			ckksTool.Eval.Rescale(ctc[i], ctc[i])

			var bootstrapErr error
			runMemoryBarrier(memoryMode.enabled, fmt.Sprintf("seqlen %d first bootstrap", t), t < 2)
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d before first bootstrap", t))
			}
			bootstrappedC, bootstrapErr := ckksTool.BootEval.Bootstrap(ctc[i])
			if bootstrapErr != nil {
				fmt.Printf("error seqlen %d first Bootstrapping faile: %v\n", t, bootstrapErr)
				return
			}
			ctc[i] = bootstrappedC
			bootstrappedC = nil
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d after first bootstrap", t))
			}

			ckksTool.Eval.Mul(ctc[i], ckksTool.ArrayToPt(w_rn[i], ctc[i].Level()), ctc[i])
			ckksTool.Eval.Rescale(ctc[i], ctc[i])

			tc := ckksTool.OptimizedFit(ctc[i], coeff.TanhC, coeff.TanhCRange)

			cto_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_3[i], thread)
			ckksTool.Eval.Add(cto_ih, ckksTool.ArrayToPt(b_ih_3[i], cto_ih.Level()), cto_ih)
			cto_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_3[i], thread)
			ckksTool.Eval.Add(cto_hh, ckksTool.ArrayToPt(b_hh_3[i], cto_hh.Level()), cto_hh)
			cto, _ := ckksTool.Eval.AddNew(cto_ih, cto_hh)
			cto_ih, cto_hh = nil, nil

			O := ckksTool.OptimizedFit(cto, coeff.Sigmoid, coeff.SigmoidRange)
			cto = nil

			if cth[i], err = ckksTool.Eval.MulRelinNew(O, tc); err != nil {
				fmt.Println(err)
			}
			O, tc = nil, nil
			ckksTool.Eval.Rescale(cth[i], cth[i])

			if i == layers-1 {
				ctx = nil
			}
			runMemoryBarrier(memoryMode.enabled, fmt.Sprintf("seqlen %d second bootstrap", t), t < 2)
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d before second bootstrap", t))
			}
			bootstrappedH, bootstrapErr := ckksTool.BootEval.Bootstrap(cth[i])
			if bootstrapErr != nil {
				fmt.Printf("error seqlen %d second Bootstrapping faile: %v\n", t, bootstrapErr)
				return
			}
			cth[i] = bootstrappedH
			bootstrappedH = nil
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d after second bootstrap", t))
			}

			if debugPrint {
				ckksTool.DecToFloat64(cth[i])
			}
		}

		ctx = nil

		if t >= minLen-1 {
			if finalH == nil {
				finalH, _ = ckksTool.Enc.EncryptNew(ptz)
			}
			clear(finalMask)
			for i := 0; i < batchSize; i++ {
				if seqLens[i] == t+1 {
					for j := 0; j < hidden_dim; j++ {
						finalMask[i+j*batchSize] = 1
					}
				}
			}
			if err := ckksTool.Eval.MulThenAdd(cth[layers-1], finalMask, finalH); err != nil {
				fmt.Printf("final hidden accumulation failed at seqlen %d: %v\n", t, err)
				return
			}
		}
		elapsed = time.Since(start)
		fmt.Printf("seqlen %d running time: %v\n", t, elapsed)

		m := logMemoryStats(fmt.Sprintf("seqlen %d end", t))
		currentAlloc := m.Alloc

		if currentAlloc > maxAlloc {
			maxAlloc = currentAlloc
		}
		if memoryMode.enabled {
			debug.FreeOSMemory()
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d after FreeOSMemory", t))
			}
		} else if t == 0 || t%10 == 0 || t == maxLen-1 {
			runtime.GC()
			if t < 2 {
				logMemoryStats(fmt.Sprintf("seqlen %d after scheduled GC", t))
			}
		}
	}
	fmt.Printf("MaxAllocated Memory (Heap): %.2f MB\n", float64(maxAlloc)/MB)

	w_fc, b_fc := params.FCWeight, utils.Repeat(batchSize, params.FCBias)

	if finalH == nil {
		finalH, _ = ckksTool.Enc.EncryptNew(ptz)
	}
	// logits := ckksTool.MatrixMultiplyWithWorkers(batchSize, finalH, w_fc, 4)

	w_fc = utils.PadMatrix(utils.Transpose(w_fc), hidden_dim, hidden_dim)
	logits := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, finalH, w_fc, thread)
	ckksTool.Eval.Add(logits, b_fc, logits)

	logRes := make([]float64, slots)
	ckksTool.End.Decode(ckksTool.Dec.DecryptNew(logits), logRes)
	utils.SaveDataToFile(logRes, fmt.Sprintf("result/%s_rms_%d.txt", dataName, batchID))

	elapsed = time.Since(startTime)
	fmt.Println("all running time: ", elapsed)

	// Compute accuracy
	labels := utils.GetLabels(dataName, batchID)
	correct := 0
	if numClasses == 1 {
		for i := 0; i < batchSize; i++ {
			predClass := 0
			if logRes[i] > 0 {
				predClass = 1
			}
			if predClass == labels[i] {
				correct++
			}
		}
	} else {
		for i := 0; i < batchSize; i++ {
			bestClass := 0
			bestVal := logRes[i] // class 0, sample i
			for j := 1; j < numClasses; j++ {
				val := logRes[j*batchSize+i]
				if val > bestVal {
					bestVal = val
					bestClass = j
				}
			}
			if bestClass == labels[i] {
				correct++
			}
		}
	}
	fmt.Printf("Accuracy: %d/%d = %.2f%%\n", correct, batchSize, float64(correct)/float64(batchSize)*100)

	file, err := os.OpenFile("elapsed.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("open file error:", err)
		return
	}
	defer file.Close()

	if _, err := file.WriteString(fmt.Sprintf("%s_%d elapsed time: %v\naccuracy: %.2f%% \n\n", dataName, batchID, elapsed, float64(correct)/float64(batchSize)*100)); err != nil {
		fmt.Println("written file error:", err)
	}

	embeddingInput = nil
	params = nil
	ckksTool = nil
	cth = nil
	ctc = nil
	finalH = nil
	logits = nil
	logRes = nil
}

func main() {
	lstm("agnews_s", 30, 64, 8)
}

// go build -o lstm_app main.go
// /usr/bin/time -v ./lstm_app
