package utils

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/kshedden/gonpy"
	"github.com/sbinet/npyio"
)

func loadNpyIntMatrix(file string) ([][]int, error) {
	f, _ := os.Open(file)
	defer f.Close()

	rdr, _ := gonpy.NewReader(f)

	data, err := rdr.GetInt64()
	if err != nil {
		return nil, err
	}

	shape := rdr.Shape
	rows, cols := shape[0], shape[1]
	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		row := make([]int, cols)
		for j := 0; j < cols; j++ {
			row[j] = int(data[i*cols+j])
		}
		matrix[i] = row
	}
	return matrix, nil
}

func loadNpyFloatMatrix(file string) ([][]float64, error) {
	f, _ := os.Open(file)
	defer f.Close()

	rdr, _ := gonpy.NewReader(f)

	data, err := rdr.GetFloat64()
	if err != nil {
		return nil, err
	}

	shape := rdr.Shape
	rows, cols := shape[0], shape[1]
	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		for j := 0; j < cols; j++ {
			row[j] = float64(data[i*cols+j])
		}
		matrix[i] = row
	}
	return matrix, nil
}

func loadNpyFloatTransposeMatrix(file string) ([][]float64, error) {
	f, _ := os.Open(file)
	defer f.Close()

	r, _ := npyio.NewReader(f)
	shape := r.Header.Descr.Shape
	m, n := shape[0], shape[1] // 原始维度 m*n

	flatData := make([]float64, m*n)
	if err := r.Read(&flatData); err != nil {
		return nil, fmt.Errorf("load data error: %w", err)
	}

	transposedMatrix := make([][]float64, n)
	for i := range transposedMatrix {
		transposedMatrix[i] = make([]float64, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			transposedMatrix[i][j] = flatData[i*m+j]
		}
	}

	return transposedMatrix, nil
}

func loadNpyIntVector(file string) ([]int, error) {
	f, _ := os.Open(file)
	defer f.Close()

	rdr, _ := gonpy.NewReader(f)

	data, err := rdr.GetInt64()
	if err != nil {
		return nil, err
	}

	vec := make([]int, len(data))
	for i, v := range data {
		vec[i] = int(v)
	}
	return vec, nil
}

func loadNpyFloatVector(file string) ([]float64, error) {
	f, _ := os.Open(file)
	defer f.Close()

	rdr, _ := gonpy.NewReader(f)
	return rdr.GetFloat64()
}

func LoadTxtFloatVector(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var result []float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if err != nil {
			return nil, err
		}
		result = append(result, val)
	}
	return result, scanner.Err()
}

// dataFile: agnews,dbpedia,imdb, 文件已经将数据按批处理，batchsize=256
func GetEmbeddings(dataFile string, batchID int) [][][]float64 {
	dataDir := "./data"
	paramsDir := "./params"

	paddedFile := filepath.Join(dataDir, dataFile, fmt.Sprintf("padded_batch_%d.npy", batchID))

	embeddingFile := filepath.Join(paramsDir, dataFile, "embedding/weight.npy")

	paddedBatch, err := loadNpyIntMatrix(paddedFile)
	if err != nil {
		log.Fatal("load padded_batch:", err)
	}

	embeddingWeight, err := loadNpyFloatMatrix(embeddingFile)
	if err != nil {
		panic(err)
	}

	batchSize := len(paddedBatch)
	seqLen := len(paddedBatch[0])
	embeddingDim := len(embeddingWeight[0])

	embeddings := make([][][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		embeddings[i] = make([][]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			tokenID := paddedBatch[i][j]
			if tokenID < 0 || tokenID >= len(embeddingWeight) {
				panic(fmt.Sprintf("token index %d out of range", tokenID))
			}
			vec := make([]float64, embeddingDim)
			copy(vec, embeddingWeight[tokenID])
			embeddings[i][j] = vec
		}
	}
	return embeddings
}

func GetLabels(dataFile string, batchID int) []int {
	dataDir := "./data"

	labelsFile := filepath.Join(dataDir, dataFile, fmt.Sprintf("labels_%d.npy", batchID))

	labels, err := loadNpyIntVector(labelsFile)
	if err != nil {
		log.Fatal("load labels:", err)
	}
	return labels
}

func GetSeqLen(dataFile string, batchID int) []int {
	dataDir := "./data"

	seqLenFile := filepath.Join(dataDir, dataFile, fmt.Sprintf("seq_lens_%d.npy", batchID))

	seqLens, err := loadNpyIntVector(seqLenFile)
	if err != nil {
		log.Fatal("load seqLens:", err)
	}
	return seqLens
}

type LSTMParams struct {
	W_ih [][]float64
	W_hh [][]float64
	B_ih []float64
	B_hh []float64
}

type LayerNormParams struct {
	Weight []float64
	Bias   []float64
}

type RMSNormParams struct {
	Weight []float64
	Bias   []float64
}

type ModelParams struct {
	LSTM map[int]LSTMParams
	// LayerNorm map[int]LayerNormParams
	RMSNorm  map[int]RMSNormParams
	FCWeight [][]float64
	FCBias   []float64
}

func GetParams(fileDir string, layers int) *ModelParams {
	paramDir := "./params"

	params := &ModelParams{
		LSTM: make(map[int]LSTMParams),
		// LayerNorm: make(map[int]LayerNormParams),
		RMSNorm: make(map[int]RMSNormParams),
	}

	for layer := 0; layer < layers; layer++ {
		l := layer

		// LSTM
		W_ih, err := loadNpyFloatMatrix(filepath.Join(paramDir, fileDir, "lstm", fmt.Sprintf("weight_ih_l%d.npy", l)))
		if err != nil {
			panic(err)
		}
		W_hh, err := loadNpyFloatMatrix(filepath.Join(paramDir, fileDir, "lstm", fmt.Sprintf("weight_hh_l%d.npy", l)))
		if err != nil {
			panic(err)
		}
		B_ih, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "lstm", fmt.Sprintf("bias_ih_l%d.npy", l)))
		if err != nil {
			panic(err)
		}
		B_hh, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "lstm", fmt.Sprintf("bias_hh_l%d.npy", l)))
		if err != nil {
			panic(err)
		}

		params.LSTM[l] = LSTMParams{
			W_ih: W_ih,
			W_hh: W_hh,
			B_ih: B_ih,
			B_hh: B_hh,
		}

		// LayerNorm
		// weight, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "layernorm", fmt.Sprintf("weight_l%d.npy", l)))
		// if err != nil {
		// 	panic(err)
		// }
		// bias, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "layernorm", fmt.Sprintf("bias_l%d.npy", l)))
		// if err != nil {
		// 	panic(err)
		// }
		// params.LayerNorm[l] = LayerNormParams{
		// 	Weight: weight,
		// 	Bias:   bias,
		// }

		// RMSNorm
		weight, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "rmsnorm", fmt.Sprintf("g_l%d.npy", l)))
		if err != nil {
			panic(err)
		}
		params.RMSNorm[l] = RMSNormParams{
			Weight: weight,
		}
		// params.LayerNorm[l] = LayerNormParams{
		// 	Weight: weight,
		// 	Bias:   bias,
		// }
	}

	// 全连接层
	// fcW, err := loadNpyFloatMatrix(filepath.Join(paramDir, fileDir, "fc", "weight.npy"))
	fcW, err := loadNpyFloatTransposeMatrix(filepath.Join(paramDir, fileDir, "fc", "weight.npy"))
	if err != nil {
		panic(err)
	}
	fcB, err := loadNpyFloatVector(filepath.Join(paramDir, fileDir, "fc", "bias.npy"))
	if err != nil {
		panic(err)
	}
	params.FCWeight = fcW
	params.FCBias = fcB

	return params
}

// 列编码
func ColCoding(x [][]float64) []float64 {
	n := len(x[0])
	p := len(x)
	res := make([]float64, p*n)
	for col := 0; col < n; col++ {
		for row := 0; row < p; row++ {
			pos := col*p + row
			res[pos] = x[row][col]
		}
	}
	return res
}

// 保存明文数据float64
func SaveDataToFile(data []float64, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, v := range data {
		_, err := f.WriteString(fmt.Sprintf("%.12f\n", v))
		if err != nil {
			return err
		}
	}
	return nil
}

func Repeat(m int, x []float64) []float64 {
	len := len(x)
	res := make([]float64, len*m)
	for i := 0; i < len; i++ {
		for j := 0; j < m; j++ {
			res[i*m+j] = x[i]
		}
	}
	return res
}

func Transpose(matrix [][]float64) [][]float64 {

	rows := len(matrix)
	cols := len(matrix[0])

	rowsT := cols
	colsT := rows

	result := make([][]float64, rowsT)
	for i := range result {
		result[i] = make([]float64, colsT)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = matrix[i][j]
		}
	}

	return result
}

func PadMatrix(input [][]float64, targetR, targetC int) [][]float64 {
	R := len(input)

	result := make([][]float64, targetR)
	for i := 0; i < targetR; i++ {
		result[i] = make([]float64, targetC)
	}

	for i := 0; i < R; i++ {
		copy(result[i], input[i])
	}

	return result
}
