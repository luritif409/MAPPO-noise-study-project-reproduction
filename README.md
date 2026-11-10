# MAPPO Observation-Noise Study

Project reproduksi terbatas R-MAPPO pada MPE `simple_spread`, dilanjutkan dengan eksplorasi terkontrol noise observasi saat evaluasi.

## Prinsip eksperimen

- Baseline dilatih pada observasi bersih.
- Checkpoint dipilih menggunakan 20 validation episode dengan seed 1001-1020.
- Checkpoint yang sama diuji pada semua tingkat noise.
- Test menggunakan seed 2001-2100 yang sama untuk setiap tingkat noise.
- Noise hanya diberikan pada fitur fisik; bagian komunikasi tidak diubah.
- Reward upstream tidak diubah. Metrik collision tambahan mengecualikan self-collision.

## Struktur

```text
config/protocol.json       protokol yang dikunci
scripts/                   urutan eksekusi 1-5
tools/                     training, validasi, evaluasi, grafik
tests/                     pemeriksaan environment dan metrik
vendor/on-policy/          upstream commit yang dipin
outputs/                   log, checkpoint, CSV, dan grafik
```

## Urutan menjalankan

### 1. Membuat environment

```bash
bash scripts/setup_env.sh
```

Project menggunakan Python 3.10, PyTorch 2.2.2 CPU, dan dependensi minimal. File `outputs/metadata/system_info.json` mencatat versi aktual dan perangkat keras. Untuk mesin CUDA, ganti baris instalasi PyTorch pada `scripts/setup_env.sh` sesuai versi CUDA mesin tersebut.

### 2. Smoke test 10.000 steps

```bash
bash scripts/01_smoke_test.sh
```

Periksa `outputs/logs/smoke_seed1.log`. Jangan lanjut jika terjadi NaN, crash, atau reward tidak tercatat.

### 3. Pilot 100.000 steps

```bash
bash scripts/02_pilot.sh
```

Pilot memeriksa apakah proses belajar mulai terlihat sebelum memakai anggaran penuh.

### 4. Baseline tiga seed

```bash
bash scripts/03_train_baselines.sh
```

Default: seed 1, 2, 3; 2 juta environment steps; 8 rollout threads; 25 langkah/episode.

### 5. Memilih checkpoint

```bash
bash scripts/04_select_checkpoints.sh
```

Checkpoint dengan validation reward tertinggi disalin ke `outputs/best/seed_X`.

### 6. Evaluasi noise

```bash
bash scripts/05_evaluate_noise.sh
```

Noise: 0; 0,05; 0,10; 0,20; 0,30. Setiap kondisi memakai 100 test episode yang sama.

### 7. Membuat learning curve

```bash
bash scripts/06_plot_training.sh
```

## Menjalankan seluruh tahap

```bash
bash scripts/run_all.sh
```

Tahap penuh dapat memakan waktu lama. Jalankan satu per satu agar hasil smoke dan pilot dapat diperiksa lebih dahulu.

## Catatan reproduksibilitas

Project mem-vendor upstream `marlbenchmark/on-policy` commit `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`. Perubahan lokal hanya:

1. menyimpan snapshot checkpoint per interval;
2. menyediakan metrik evaluasi `collisions`, `min_dists`, dan `occupied_landmarks`;
3. meneruskan seluruh diagnostics skenario melalui `info` environment;
4. menambahkan pipeline validation dan test-time observation noise.



