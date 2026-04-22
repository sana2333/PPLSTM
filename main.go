package main

import (
	"fmt"
	"lstm/ckkstool"
	"lstm/coeffs"
	"lstm/utils"
	"os"
	"runtime"
	"slices"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const MB = 1024 * 1024

func lstm(dataName string, batchID int, hidden_dim int, thread int) {
	var start time.Time
	var elapsed time.Duration

	var maxAlloc uint64

	fmt.Println("ckks initting")
	start = time.Now()
	ckksTool, err := ckkstool.NewCKKSTool()
	if err != nil {
		fmt.Println(err)
	} else {
		elapsed = time.Since(start)
		fmt.Println("ckks init success and the running time: ", elapsed)
	}

	slots := ckksTool.Params.MaxSlots()

	embeddings := utils.GetEmbeddings(dataName, batchID)
	batchSize := len(embeddings)
	seqLen := len(embeddings[0])
	fmt.Println(len(embeddings[0][0]), " ", seqLen, " ", batchSize)
	seqLens := utils.GetSeqLen(dataName, batchID)
	minLen := slices.Min(seqLens)
	maxLen := slices.Max(seqLens)
	fmt.Println(maxLen)

	layers := 1
	params := utils.GetParams(dataName, layers)

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

	finalH, _ := ckksTool.Enc.EncryptNew(ptz)

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

	startTime := time.Now()
	for t := 0; t < maxLen; t++ {
		fmt.Printf("\n start seqlen %d ===\n", t)
		start = time.Now()
		x := make([][]float64, batchSize)
		for i := 0; i < batchSize; i++ {
			x[i] = embeddings[i][t]
		}
		ptx := ckks.NewPlaintext(ckksTool.Params, ckksTool.Params.MaxLevel())
		xcol := utils.ColCoding(utils.PadMatrix(x, batchSize, hidden_dim))
		ckksTool.End.Encode(xcol, ptx)
		ctx, _ := ckksTool.Enc.EncryptNew(ptx)

		defer func() {
			x = nil
			xcol = nil
			ptx = nil
		}()

		for i := 0; i < layers; i++ {
			cti_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_0[i], thread)
			ckksTool.Eval.Add(cti_ih, ckksTool.ArrayToPt(b_ih_0[i], cti_ih.Level()), cti_ih)
			cti_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_0[i], thread)
			ckksTool.Eval.Add(cti_hh, ckksTool.ArrayToPt(b_hh_0[i], cti_hh.Level()), cti_hh)
			cti, _ := ckksTool.Eval.AddNew(cti_ih, cti_hh)

			ctf_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_1[i], thread)
			ckksTool.Eval.Add(ctf_ih, ckksTool.ArrayToPt(b_ih_1[i], ctf_ih.Level()), ctf_ih)
			ctf_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_1[i], thread)
			ckksTool.Eval.Add(ctf_hh, ckksTool.ArrayToPt(b_hh_1[i], ctf_hh.Level()), ctf_hh)
			ctf, _ := ckksTool.Eval.AddNew(ctf_ih, ctf_hh)

			ctg_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_2[i], thread)
			ckksTool.Eval.Add(ctg_ih, ckksTool.ArrayToPt(b_ih_2[i], ctg_ih.Level()), ctg_ih)
			ctg_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_2[i], thread)
			ckksTool.Eval.Add(ctg_hh, ckksTool.ArrayToPt(b_hh_2[i], ctg_hh.Level()), ctg_hh)
			ctg, _ := ckksTool.Eval.AddNew(ctg_ih, ctg_hh)

			cto_ih := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, ctx, w_ih_3[i], thread)
			ckksTool.Eval.Add(cto_ih, ckksTool.ArrayToPt(b_ih_3[i], cto_ih.Level()), cto_ih)
			cto_hh := ckksTool.MatrixMultiplyPCMMDiagonalBSGS(batchSize, cth[i], w_hh_3[i], thread)
			ckksTool.Eval.Add(cto_hh, ckksTool.ArrayToPt(b_hh_3[i], cto_hh.Level()), cto_hh)
			cto, _ := ckksTool.Eval.AddNew(cto_ih, cto_hh)

			I := ckksTool.OptimizedFit(cti, coeff.Sigmoid, coeff.SigmoidRange)
			F := ckksTool.OptimizedFit(ctf, coeff.Sigmoid, coeff.SigmoidRange)
			G := ckksTool.OptimizedFit(ctg, coeff.TanhG, coeff.TanhGRange)
			O := ckksTool.OptimizedFit(cto, coeff.Sigmoid, coeff.SigmoidRange)

			ckksTool.Eval.MulRelin(ctc[i], F, ctc[i])
			ckksTool.Eval.Rescale(ctc[i], ctc[i])

			temp, err := ckksTool.Eval.MulRelinNew(I, G)
			if err != nil {
				fmt.Println(err)
			}
			ckksTool.Eval.Rescale(temp, temp)
			ckksTool.Eval.Add(ctc[i], temp, ctc[i])

			ms := ckksTool.MeanSquare(batchSize, hidden_dim, ctc[i])
			ckksTool.Eval.Add(ms, 1e-5, ms)

			rv := ckksTool.OptimizedFit(ms, coeff.Rsqrt, coeff.RsqrtRange)
			ckksTool.Eval.MulRelin(ctc[i], rv, ctc[i])

			ckksTool.Eval.Rescale(ctc[i], ctc[i])

			var bootstrapErr error
			if ctc[i], bootstrapErr = ckksTool.BootEval.Bootstrap(ctc[i]); bootstrapErr != nil {
				fmt.Printf("error seqlen %d first Bootstrapping faile: %v\n", t, bootstrapErr)
				return
			}

			ckksTool.Eval.Mul(ctc[i], ckksTool.ArrayToPt(w_rn[i], ctc[i].Level()), ctc[i])
			ckksTool.Eval.Rescale(ctc[i], ctc[i])

			tc := ckksTool.OptimizedFit(ctc[i], coeff.TanhC, coeff.TanhCRange)

			if cth[i], err = ckksTool.Eval.MulRelinNew(O, tc); err != nil {
				fmt.Println(err)
			}
			ckksTool.Eval.Rescale(cth[i], cth[i])

			if cth[i], bootstrapErr = ckksTool.BootEval.Bootstrap(cth[i]); bootstrapErr != nil {
				fmt.Printf("error seqlen %d second Bootstrapping faile: %v\n", t, bootstrapErr)
				return
			}
		}

		if t >= minLen-1 {
			mask := make([]float64, slots)
			for i := 0; i < batchSize; i++ {
				if seqLens[i] == t+1 {
					for j := 0; j < hidden_dim; j++ {
						mask[i+j*batchSize] = 1
					}
				}
			}
			ckksTool.Eval.MulThenAdd(cth[layers-1], mask, finalH)
		}
		elapsed = time.Since(start)
		fmt.Printf("seqlen %d running time: %v\n", t, elapsed)

		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		currentAlloc := m.Alloc
		currentSys := m.Sys
		fmt.Printf("seqlen %d: Current Allocated Memory (Heap): %.2f MB\n", t, float64(currentAlloc)/MB)
		fmt.Printf("seqlen %d: Current System Memory (Heap): %.2f MB\n", t, float64(currentSys)/MB)

		if currentAlloc > maxAlloc {
			maxAlloc = currentAlloc
		}
	}
	fmt.Printf("MaxAllocated Memory (Heap): %.2f MB\n", float64(maxAlloc)/MB)

	finalRes := make([]float64, slots)
	ckksTool.End.Decode(ckksTool.Dec.DecryptNew(finalH), finalRes)
	// utils.SaveDataToFile(finalRes, fmt.Sprintf("result/%s_rms_h_%d.txt", dataName, batchID))

	w_fc, b_fc := params.FCWeight, utils.Repeat(batchSize, params.FCBias)
	logits := ckksTool.MatrixMultiplyWithWorkers(batchSize, finalH, w_fc, 4)
	ckksTool.Eval.Add(logits, b_fc, logits)

	logRes := make([]float64, slots)
	ckksTool.End.Decode(ckksTool.Dec.DecryptNew(logits), logRes)
	utils.SaveDataToFile(logRes, fmt.Sprintf("result/%s_rms_s_%d.txt", dataName, batchID))

	elapsed = time.Since(startTime)
	fmt.Println("all running time: ", elapsed)

	file, err := os.OpenFile("elapsed.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("open file error:", err)
		return
	}
	defer file.Close()

	if _, err := file.WriteString(fmt.Sprintf("%s_%d elapsed time: %v\n", dataName, batchID, elapsed)); err != nil {
		fmt.Println("written file error:", err)
	}
}

func main() {
	lstm("yelp", 150, 128, 8)
}
