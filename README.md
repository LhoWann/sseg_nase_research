# SSEG-NASE Pipeline Usage Guide

## 1. Instalasi & Persiapan

- Pastikan Python 3.9+ dan CUDA sudah terpasang.
- Install semua dependensi:
  ```bash
  pip install -r requirements.txt
  ```
- Siapkan dataset pada folder `datasets/` (minimagenet, cifar_fs, dsb).

---

## 2. Menjalankan Full Pipeline Otomatis

Script utama: `scripts_cli_run.py` (pipeline otomatis: curriculum, training, evaluasi, ablation, analisis, export)

### Contoh Perintah:

```bash
python scripts_cli_run.py configs_experiments_exp_tuned_pipeline.json
```

- Semua tahapan (generate curriculum, train, evaluate, ablation, analyze, export) dijalankan otomatis sesuai config JSON.
- Output, checkpoint, dan log tersimpan di folder `outputs/tuned_pipeline/`.

---

## 3. Menjalankan Script Individual (Manual)

### a. Generate Curriculum

```bash
python scripts/generate_curriculum.py --config configs/experiments/exp_tuned_pipeline.yaml --output-dir outputs/tuned_pipeline/synthetic_curriculum
```

### b. Training

```bash
python scripts/train_sseg.py --config configs/experiments/exp_tuned_pipeline.yaml --output-dir outputs/tuned_pipeline/sseg_nase_tuned_pipeline --seed 42 --device cuda
```

### c. Evaluasi Few-Shot

```bash
python scripts/evaluate_fewshot.py \
  --checkpoint outputs/tuned_pipeline/sseg_nase_tuned_pipeline/checkpoints/final_model.pt \
  --data-dir datasets/minimagenet \
  --output-dir outputs/tuned_pipeline/results \
  --num-ways 5 \
  --num-shots 1 5 \
  --num-episodes 600 \
  --seed 42 \
  --device cuda \
  --config configs/experiments/exp_tuned_pipeline.yaml
```

### d. Ablation Study

```bash
python scripts/run_ablation.py --config configs/experiments/exp_tuned_pipeline.yaml --output-dir outputs/tuned_pipeline/ablation --seed 42 --device cuda
```

### e. Analisis Evolusi

```bash
python scripts/analyze_evolution.py --log-dir outputs/tuned_pipeline/sseg_nase_tuned_pipeline/logs --output-dir outputs/tuned_pipeline/analysis
```

### f. Export Hasil

```bash
python scripts/export_results.py --input-dir outputs/tuned_pipeline/results --format latex markdown json csv --output-dir outputs/tuned_pipeline/export
```

---

## 4. Notebook Interaktif

- Untuk eksplorasi data, visualisasi arsitektur, dan analisis hasil, gunakan notebook di folder `notebooks/`.

---

## 5. Troubleshooting

- Jika terjadi error OOM, kecilkan batch_size atau image_size di config.
- Jika terjadi size mismatch saat evaluasi, pastikan checkpoint dan config sesuai.
- Semua log dan error tercatat di folder `outputs/tuned_pipeline/sseg_nase_tuned_pipeline/logs`.

---

## 6. Struktur Output

- `outputs/tuned_pipeline/sseg_nase_tuned_pipeline/checkpoints/` : Model checkpoint
- `outputs/tuned_pipeline/sseg_nase_tuned_pipeline/logs/` : Log training & evolusi
- `outputs/tuned_pipeline/results/` : Hasil evaluasi few-shot
- `outputs/tuned_pipeline/ablation/` : Hasil ablation
- `outputs/tuned_pipeline/analysis/` : Analisis evolusi
- `outputs/tuned_pipeline/export/` : Export hasil (latex, markdown, json, csv)

---

## 7. Reproducibilitas

- Set seed di config untuk eksperimen deterministik.
- Semua setting dan hasil otomatis terdokumentasi di outputs.
