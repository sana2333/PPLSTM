# PPLSTM Reproducibility Guide

This repository runs encrypted LSTM inference (CKKS, via Lattigo) on preprocessed text batches.
The steps below are written so results can be reproduced and independently checked.

## 1. Environment Setup

1. Install Go (project uses `go 1.23.0`, toolchain `go1.24.8` in `go.mod`).
2. Clone this repository and enter the project root:

```bash
git clone https://github.com/sana2333/PPLSTM.git
cd PPLSTM
```

3. Download model checkpoints from the repository **Releases** page. Please extract the files to the project's root directory.

## 2. Dependencies

Install dependencies from `go.mod`:

```bash
go mod download
```

Main libraries used by this project:

- `github.com/tuneinsight/lattigo/v6` (CKKS/FHE)
- `github.com/kshedden/gonpy` and `github.com/sbinet/npyio` (read `.npy`)

## 3. Data and Parameter Layout

`data/` is already included in this repo. Parameters from Releases must match this structure:

```text
params/
  <dataset>/
    embedding/weight.npy
    lstm/weight_ih_l0.npy
    lstm/weight_hh_l0.npy
    lstm/bias_ih_l0.npy
    lstm/bias_hh_l0.npy
    rmsnorm/g_l0.npy
    fc/weight.npy
    fc/bias.npy
```

Supported `dataset` values (from code):

- `agnews`, `agnews_s`, `dbpedia`, `dbpedia_s`, `imdb`, `imdb_s`, `yelp`, `yelp_s`

Supported `TokensID` values (from `data/` files):

- `agnews`, `agnews_s`, `dbpedia`, `dbpedia_s`: `30`, `50`, `70`
- `imdb`, `imdb_s`, `yelp`, `yelp_s`: `100`, `150`, `200`

## 4. Important Path Consistency Note

Current code reads parameter files from two paths:

- `./params` (embedding in `GetEmbeddings`)
- `../params` (LSTM/RMSNorm/FC in `GetParams`)

To avoid file-not-found errors, choose one method:

1. Recommended: keep `params/` in project root and make paths consistent in code (`utils/utils.go`, `GetParams`, set `paramDir := "./params"`).
2. No-code-change workaround: keep `params/` in project root, and also create `../params` as a symlink (or copy) pointing to the same folder.

## 5. Configure Experiment Parameters

Edit [`main.go`](/D:/Code/golang/PPLSTM/main.go):

1. Set dataset and tokens in `main()`:
   
   ```go
   lstm("yelp", 150, 128, 8)
   ```

2. Set hidden dimension in `lstm()`:
- standard models (`agnews`, `dbpedia`, `imdb`, `yelp`): `hidden_dim := 128`
- `_s` models (`agnews_s`, `dbpedia_s`, `imdb_s`, `yelp_s`): `hidden_dim := 64`
3. Optional runtime parameters:
- `length := 150` (input tokens)
- `thread := 8` (matrix multiply worker threads)
4. CKKS parameters:
   
   ```go
   params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
       LogN:            15,
       LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
       LogP:            []int{61, 61, 61},
       LogDefaultScale: 40,
       RingType:        ring.ConjugateInvariant,
       Xs:              ring.Ternary{H: 192},
   })
   
   bootParams, _ := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
       LogN:     utils.Pointy(16),
       LogP:     []int{61, 61, 61, 61},
       Xs:       params.Xs(),
       LogSlots: utils.Pointy(15),
   })
   ```

5. Polynomial coefficients:
   
   The polynomial approximation coefficients used in the privacy-preserving LSTM inference are defined in the `coeff.go` file.
   
   These coefficients are used to approximate the nonlinear activation functions and normalization operations required during encrypted inference, including:
   
   - `Sigmoid`
   - `TanhG` (for gate activations)
   - `TanhC` (for cell state output)
   - `Rsqrt` (for RMSNorm)
   
   An example configuration is shown below:
   
   ```go
   "agnews": {
   		Sigmoid: []float64{5.02344327e-01, 2.08822638e-01, -8.23101112e-04, -6.94408215e-03,
   			3.92880069e-05, 1.19941125e-04, -4.43725000e-07, -7.29902420e-07},
   
   		TanhG: []float64{7.01964345e-03, 6.31479643e-01, -1.09247082e-03, -3.82442277e-02,
   			-2.15438964e-04, 1.02211252e-03, 1.15061130e-05, -8.04698511e-06},
   
   		Rsqrt: []float64{2.58685626, -3.77715572, 2.79941214, -0.70589577},
   
   		TanhC: []float64{2.83421535e-05, 6.34632931e-01, -5.03537882e-05, -3.34134785e-02,
   			6.77925704e-06, 7.31655135e-04, -1.62566023e-07, -5.14137210e-06},
   
   		SigmoidRange: [2]float64{-9, 8.5},
   		TanhGRange:   [2]float64{-7.5, 5.5},
   		RsqrtRange:   [2]float64{0.01, 2},
   		TanhCRange:   [2]float64{-7.5, 8.5},
   	}
   ```
   
   All polynomial coefficients are stored **from the lowest degree term to the highest degree term**. For example, the `Rsqrt` coefficients represent the polynomial:
   
   $$
   Rsqrt(x)=2.58685626+(-3.77715572)*x+2.79941214*x^2+-0.70589577)*x^3
   $$
   
   This ordering is consistently used throughout `coeff.go` for all activation approximations.
   
   Each polynomial is associated with a predefined approximation interval, specified by its corresponding `Range` field. For example, the `Rsqrt` polynomial approximation is optimized over the interval **[0.01, 2]**. Similarly, the ranges for `Sigmoid`, `TanhG`, and `TanhC` define the valid approximation domains for their respective functions.
   
   

## 6. Run

From repository root:

```bash
go run main.go
```

## 7. Outputs and Verification

After each run, check:

1. Logits output file:
- `result/<dataset>_rms_s_<input tokens>.txt`
2. Runtime log:
- `elapsed.txt` (appended each run, format: `<dataset>_<input tokens> elapsed time: ...`)
3. Independent verification checklist:
- confirm the selected `dataset`, `input tokens`, `hidden_dim`, and `threads` match
- confirm parameter files were loaded from the expected `params` path(s)
- confirm output file exists and has non-empty float values
