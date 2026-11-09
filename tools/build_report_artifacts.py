"""Build source-backed, read-only audit artifacts for the MAPPO noise study.

This script only writes four report files under reports/. It never runs training or
changes experiment outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
EP = pd.read_csv(ROOT / "outputs/evaluation/noise_episode_results.csv")
PER = pd.read_csv(ROOT / "outputs/evaluation/noise_results_per_training_seed.csv")
SUM = pd.read_csv(ROOT / "outputs/evaluation/noise_summary.csv")
PROTOCOL = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))


def fmt(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


validation = []
for seed in (1, 2, 3):
    e = pd.read_csv(ROOT / f"outputs/validation/seed_{seed}_episodes.csv")
    s = pd.read_csv(ROOT / f"outputs/validation/seed_{seed}_summary.csv")
    j = json.loads((ROOT / f"outputs/best/seed_{seed}/selection.json").read_text())
    chosen = Path(j["selected_checkpoint"])
    copied = ROOT / f"outputs/best/seed_{seed}/actor.pt"
    validation.append(dict(seed=seed, episodes=len(e), candidates=len(s),
                           seeds=sorted(e.episode_seed.unique().tolist()),
                           every_20=all(e.groupby("checkpoint").episode_seed.nunique() == 20),
                           selected_is_top=str(s.iloc[0].checkpoint) == str(chosen),
                           checkpoint_exists=chosen.exists() and copied.exists(),
                           copy_identical=chosen.exists() and copied.exists() and sha(chosen) == sha(copied),
                           selected=chosen.name, mean=j["mean_reward"]))

checks = [
    ("Baris episode", "1.500", str(len(EP)), len(EP) == 1500),
    ("Baris per training seed", "15", str(len(PER)), len(PER) == 15),
    ("Baris ringkasan", "5", str(len(SUM)), len(SUM) == 5),
    ("Training seed", "1, 2, 3", ", ".join(map(str, sorted(EP.train_seed.unique()))), set(EP.train_seed) == {1,2,3}),
    ("Tingkat noise", "0; 0,05; 0,10; 0,20; 0,30", "; ".join(fmt(x,2) for x in sorted(EP.noise_std.unique())), set(np.round(EP.noise_std,2)) == {0,.05,.1,.2,.3}),
    ("Episode per sel", "100", f"{EP.groupby(['train_seed','noise_std']).size().min()}–{EP.groupby(['train_seed','noise_std']).size().max()}", EP.groupby(['train_seed','noise_std']).size().eq(100).all()),
    ("Test seed", "2001–2100 per sel", f"{EP.episode_seed.min()}–{EP.episode_seed.max()}", all(set(g.episode_seed)==set(range(2001,2101)) for _,g in EP.groupby(['train_seed','noise_std']))),
    ("Nilai kosong / NaN / infinit", "0", str(int(EP.isna().sum().sum())+int(np.isinf(EP.select_dtypes('number').to_numpy()).sum())), not EP.isna().any().any() and np.isfinite(EP.select_dtypes('number').to_numpy()).all()),
    ("Duplikasi kunci", "0", str(int(EP.duplicated(['train_seed','noise_std','episode_seed']).sum())), not EP.duplicated(['train_seed','noise_std','episode_seed']).any()),
    ("Horizon", "25", str(EP.episode_length.unique().tolist()), EP.episode_length.eq(25).all()),
    ("Rentang success", "0 atau 1", str(sorted(EP.success.unique())), EP.success.isin([0,1]).all()),
    ("Collision dan jarak", "bilangan hingga, tidak negatif", f"min {EP.collision_pairs.min():.0f}; {EP.final_min_dists.min():.3f}", (EP.collision_pairs.ge(0).all() and EP.final_min_dists.ge(0).all())),
    ("Landmark terisi", "bilangan bulat 0–3", str(sorted(EP.final_occupied_landmarks.unique())), EP.final_occupied_landmarks.isin([0,1,2,3]).all()),
    ("Tipe kolom", "integer dan numerik sesuai skema", ", ".join(f"{k}:{v}" for k,v in EP.dtypes.items()), all(pd.api.types.is_numeric_dtype(v) for v in EP.dtypes)),
    ("Validasi checkpoint", "seed 1001–1020", "3 berkas episode", all(v['seeds']==list(range(1001,1021)) and v['every_20'] for v in validation)),
    ("Checkpoint terpilih", "ada dan salinan identik", "3 seed", all(v['selected_is_top'] and v['checkpoint_exists'] and v['copy_identical'] for v in validation)),
]

recomputed = EP.groupby(['train_seed','noise_std'],as_index=False).agg(mean_reward=('episode_reward','mean'),success_rate=('success','mean'),mean_collisions=('collision_pairs','mean'),mean_final_distance=('final_min_dists','mean'))
aggregate_error = max((recomputed[c]-PER[c]).abs().max() for c in ['mean_reward','success_rate','mean_collisions','mean_final_distance'])
checks.append(("Agregat per seed", "sama dengan CSV episode", f"galat maks. {aggregate_error:.2e}", aggregate_error < 1e-10))
summary_error = max(abs(PER.groupby('noise_std')[c].mean().to_numpy()-SUM[d].to_numpy()).max() for c,d in [('mean_reward','reward_mean'),('success_rate','success_mean'),('mean_collisions','collisions_mean')])
checks.append(("Ringkasan antarseed", "sama dengan CSV per seed", f"galat maks. {summary_error:.2e}", summary_error < 1e-10))

dist = PER.groupby('noise_std').mean_final_distance.agg(['mean','std']).reset_index()
main = SUM.merge(dist, on='noise_std').rename(columns={'mean':'distance_mean','std':'distance_std'})
base = main.iloc[0]
for col in ['reward_mean','success_mean','collisions_mean','distance_mean']:
    main[f'delta_{col}'] = main[col] - base[col]

# A CSV is selected explicitly by the user and keeps typed values with clear provenance.
main.to_csv(OUT / 'Tabel_Hasil_Utama.csv', index=False)

audit_paths = [
    ('README.md','Panduan proyek'),('CODE_MAP.md','Peta sistem dan kode'),('REPORT_TEMPLATE.md','Kerangka laporan'),
    ('config/protocol.json','Protokol eksperimen'),('tools/run_training.py','Peluncur training'),
    ('tools/select_checkpoint.py','Pemilihan checkpoint'),('tools/evaluate_noise.py','Agregasi evaluasi'),('tools/eval_core.py','Noise dan metrik episode'),
    ('outputs/metadata/system_info.json','Metadata sistem'),
]
for i in (1,2,3):
    audit_paths += [(f'outputs/runs/baseline_seed{i}.json',f'Metadata training seed {i}'),
                    (f'outputs/best/seed_{i}/selection.json',f'Pilihan checkpoint seed {i}')]
audit_paths += [('outputs/validation/','Episode dan peringkat validasi'),
                ('outputs/evaluation/noise_episode_results.csv','Data episode test'),
                ('outputs/evaluation/noise_results_per_training_seed.csv','Ringkasan per seed'),
                ('outputs/evaluation/noise_summary.csv','Ringkasan antarseed')]
for name in ['baseline_learning_curves.png','noise_vs_reward.png','noise_vs_success.png','noise_vs_collisions.png']:
    audit_paths.append((f'outputs/figures/{name}','Visualisasi hasil'))
audit_paths.append(('references/','Dua paper rujukan'))

issues = {'outputs/metadata/system_info.json':'Python 3.13.2 dan torch null; tidak cocok dengan README dan jalur training.',
          'README.md':'Menyebut Python 3.10; metadata sistem berbeda.',
          'outputs/validation/':'Periksa ringkasan dan seed; validasi tidak dapat menggantikan test.',
          'references/':'Dua paper tersedia; BenchMARL bukan paper eksperimen noise ini.'}
audit = ['# Ringkasan Audit Eksperimen', '', 'Audit baca saja terhadap artefak yang tersedia pada 9 November 2026. Hasil di `outputs/verification_1000_steps` dikecualikan.', '',
         '## Tabel audit berkas', '', '| Berkas | Fungsi | Status | Masalah/catatan |', '|---|---|---|---|']
for path, purpose in audit_paths:
    exists=(ROOT/path).exists()
    audit.append(f'| `{path}` | {purpose} | {"Tersedia" if exists else "Tidak tersedia"} | {issues.get(path,"—" if exists else "Bukti tidak tersedia") } |')
audit += ['', '## Validasi data', '', '| Pemeriksaan | Harapan | Temuan | Status |','|---|---:|---:|---|']
for label, expected, found, ok in checks:
    audit.append(f'| {label} | {expected} | {found} | {"Lulus" if ok else "Gagal"} |')
audit += ['', '## Catatan yang memengaruhi interpretasi', '',
          '- Seluruh 1.500 nilai `success` adalah nol. Definisi success ialah ketiga landmark ditempati pada langkah terakhir dengan ambang jarak <0,1; ini menunjukkan floor effect dan tidak membedakan tingkat noise.',
          '- `collision_pairs` adalah jumlah pasangan tabrakan sepanjang episode; reward upstream tetap menghitung self-collision. Oleh sebab itu tren keduanya tidak wajib sama.',
          '- Metadata sistem mencatat `torch: null` dan Python 3.13.2, sedangkan README menyebut Python 3.10; jalur executable dalam metadata berbeda dari perintah run. Versi Python/PyTorch saat training tidak dapat dipastikan dari berkas metadata ini.',
          '- `outputs/logs/smoke_seed1.log` berakhir dengan `KeyboardInterrupt`, meski `outputs/runs/smoke_seed1.json` tersedia. Tidak ada metadata pilot pada `outputs/runs/`; tahap pilot tidak dapat diverifikasi sebagai selesai.',
          '- Hanya tiga training seed, satu skenario, dan 2 juta steps per baseline. Tidak ada uji signifikansi; efek kecil tidak boleh dianggap bukti kausal yang kuat.',
          '', '## Checkpoint terpilih', '', '| Seed | Snapshot | Kandidat | Episode validasi | Reward validasi | Salinan identik |','|---:|---|---:|---:|---:|---|']
for v in validation:
    audit.append(f'| {v["seed"]} | `{v["selected"]}` | {v["candidates"]} | {v["episodes"]} | {fmt(v["mean"])} | {"Ya" if v["copy_identical"] else "Tidak"} |')
audit += ['', '## Sumber hitung', '',
          '- `outputs/evaluation/noise_episode_results.csv`: kunci, nilai episode, dan rentang.',
          '- `outputs/evaluation/noise_results_per_training_seed.csv`: rerata setiap training seed.',
          '- `outputs/evaluation/noise_summary.csv`: rerata dan simpangan baku sampel antarseed.',
          '- `tools/evaluate_noise.py`: definisi agregasi dan persentase degradasi reward.',
          '- `tools/eval_core.py` dan `vendor/on-policy/onpolicy/envs/mpe/scenarios/simple_spread.py`: definisi noise dan metrik.',
          '']
(OUT/'Ringkasan_Audit_Eksperimen.md').write_text('\n'.join(audit),encoding='utf-8')

doc=Document()
sec=doc.sections[0]
sec.top_margin=Cm(2.4); sec.bottom_margin=Cm(2.2); sec.left_margin=Cm(2.7); sec.right_margin=Cm(2.7)
styles=doc.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(10)
styles['Normal'].paragraph_format.space_after=Pt(5)
for n,size in [('Title',18),('Heading 1',14),('Heading 2',11)]:
    styles[n].font.name='Aptos'; styles[n].font.size=Pt(size); styles[n].font.color.rgb=RGBColor(0,0,0)
title_ppr=styles['Title']._element.pPr
for border in title_ppr.findall(qn('w:pBdr')):
    title_ppr.remove(border)

def p(t='',style=None):
    return doc.add_paragraph(t,style)
def h(t,level=1):
    heading=doc.add_heading(t,level)
    heading.paragraph_format.keep_with_next=True
def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,x in enumerate(headers): t.rows[0].cells[i].text=str(x)
    for row in rows:
        cells=t.add_row().cells
        for i,x in enumerate(row): cells[i].text=str(x)
    for ri,row in enumerate(t.rows):
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(0)
                for run in para.runs: run.font.size=Pt(8.5)
            if ri==0:
                tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'DCE6F1'); tcPr.append(shd)
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next=True
                    for run in para.runs: run.bold=True
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    doc.add_paragraph()
def fig(name,cap):
    path=ROOT/'outputs/figures'/name
    if path.exists():
        para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(path),width=Cm(14.3))
        q=p(cap); q.alignment=WD_ALIGN_PARAGRAPH.CENTER

p('Laporan Reproduksi Terbatas MAPPO dan Ketahanan terhadap Noise Observasi','Title')
p('MPE simple_spread · Metodologi DEGREE · Audit artefak 9 November 2026')
p('Sumber: artefak eksperimen lokal dan dua paper pada folder references. Penulis/afiliasi tidak dicantumkan karena tidak tersedia pada proyek.')
doc.add_page_break()
h('ABSTRAK')
p(f'Penelitian ini mengaudit reproduksi terbatas R-MAPPO pada MPE simple_spread dan mengevaluasi gangguan Gaussian pada observasi fisik saat pengujian. Tiga baseline dilatih dengan seed 1–3 dan anggaran 2 juta environment steps per seed. Checkpoint dipilih pada seed validasi 1001–1020; setiap checkpoint diuji pada lima tingkat noise dengan 100 episode test berseed 2001–2100, menghasilkan 1.500 baris lengkap. Rerata reward bergerak dari {fmt(base.reward_mean)} pada σ=0 menjadi {fmt(main.iloc[-1].reward_mean)} pada σ=0,30; degradasi relatif {fmt(main.iloc[-1].reward_degradation_pct)}%. Seluruh success rate bernilai nol, sehingga metrik tersebut mengalami floor effect. Penurunan reward tidak monoton karena pada σ=0,05 rerata sedikit membaik. Bukti terbatas pada satu skenario, tiga training seed, dan metadata lingkungan yang tidak konsisten; temuan tidak dapat digeneralisasi sebagai reproduksi penuh atau efek signifikan secara statistik.')
h('1. Pendahuluan')
h('1.1 Latar belakang',2); p('Koordinasi multiagen pada MPE mengandalkan observasi lokal yang dapat terganggu. Paper MAPPO menunjukkan PPO kompetitif pada tugas kooperatif, tetapi tidak menguji desain noise observasi ini [1].')
h('1.2 Rumusan masalah',2); p('Bagaimana kinerja checkpoint R-MAPPO yang dilatih pada observasi bersih berubah ketika fitur fisik observasi diberi noise Gaussian saat evaluasi?')
h('1.3 Tujuan',2); p('Mengukur reward, keberhasilan, tabrakan, dan jarak akhir untuk lima kondisi noise serta memeriksa keterlacakan dan konsistensi seluruh artefak.')
h('1.4 Batasan',2); p('Satu skenario simple_spread; tiga seed training; tanpa retraining untuk kondisi noise; tanpa uji signifikansi atau klaim generalisasi lintas lingkungan.')
h('2. Review Paper')
h('2.1 Paper pertama',2)
p('Yu, Velu, Vinitsky, Gao, Wang, Bayen, dan Wu, “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” NeurIPS 2022 [1]. Tujuannya mengevaluasi PPO kooperatif terhadap metode off-policy dan mengidentifikasi faktor implementasi. Masalah utamanya ialah anggapan bahwa PPO kurang efisien sampel di MARL. Lingkungan: MPE, SMAC, Google Research Football, dan Hanabi. MAPPO menggunakan actor berbasis observasi lokal dan critic berbasis informasi global; input berupa observasi/state, output berupa aksi, nilai, dan kinerja. Eksperimen MPE mencakup Spread, Reference, dan Comm dengan 10 seed; metrik mencakup return dan metrik spesifik testbed. Temuan: MAPPO sering sebanding atau lebih baik daripada pembanding dengan efisiensi sampel yang kompetitif. Kelebihan: perbandingan luas dan ablation. Keterbatasan untuk studi ini: tidak menguji noise observasi evaluasi dan tidak menyediakan pembuktian ketahanan pada konfigurasi lokal.')
h('2.2 Paper kedua',2)
p('Bettini, Prorok, dan Moens, “BenchMARL: Benchmarking Multi-Agent Reinforcement Learning,” Journal of Machine Learning Research 25 (2024), artikel 23-1612 [2]. Tujuannya menstandarkan benchmark MARL untuk mengurangi kesulitan reproduksibilitas. Solusinya ialah pustaka berbasis TorchRL dengan konfigurasi terstruktur, eksperimen, tugas, algoritma, dan model. Lingkungan mencakup VMAS, SMACv2, MPE, SISL, dan MeltingPot; algoritma mencakup MAPPO, IPPO, MADDPG, dan lainnya. Input: konfigurasi algoritma, model, tugas, dan seed; output: run, log, serta plot benchmark. Hasil contoh VMAS menggunakan normalized return, kurva efisiensi sampel, profil performa, dan skor agregat. Kelebihan: standar konfigurasi dan pelaporan. Keterbatasan: bukan implementasi yang dipakai proyek ini dan tidak memvalidasi angka eksperimen noise lokal.')
h('2.3 Sintesis dan kesenjangan',2); p('Paper pertama mendasari MAPPO dan pembanding; paper kedua menekankan standar reproduksibilitas. Eksperimen lokal menambah uji noise saat evaluasi dan membutuhkan audit metadata, seed, serta ringkasan antarseed. Kedua paper tidak menjadi bukti langsung nilai hasil lokal.')
h('3. Pemahaman Sistem dan Kode')
h('3.1 MPE simple_spread',2); p('Tiga agen bergerak menuju tiga landmark. Observasi mencakup kecepatan dan posisi sendiri, posisi relatif landmark dan agen lain, serta kanal komunikasi. Skenario menetapkan reward negatif dari jumlah jarak minimum landmark ke agen dan penalti collision; reward sumber tetap menghitung self-collision. Metrik collision evaluasi menghapus self-collision [3,4].')
h('3.2 MAPPO',2); p('Alur: lingkungan → observasi lokal agen → actor → aksi → reward → trajectory → GAE/advantage → objective PPO → pembaruan actor dan critic → evaluasi. Runner menyimpan rollout sepanjang 25 langkah; critic terpusat memakai gabungan observasi agen selama training [5,6].')
h('3.3 Actor dan centralized critic',2); p('Actor R-MAPPO memetakan observasi lokal ke distribusi aksi. Critic memetakan share_obs ke nilai keadaan saat training. Pada evaluasi, hanya actor checkpoint terpilih dimuat; aksi dipilih deterministik [4,7].')
h('3.4 Objective PPO dan GAE',2)
p('r_t(θ) = π_θ(a_t|o_t) / π_θ_old(a_t|o_t). θ ialah parameter actor baru, θ_old parameter saat rollout, a_t aksi, o_t observasi, dan r_t rasio probabilitas. Dalam kode, rasio dihitung sebagai exp(log π baru − log π lama) [8, baris 129].')
p('L_clip = E[min(r_t A_t, clip(r_t, 1−ε, 1+ε)A_t)]. A_t ialah advantage yang diperoleh dari return dan prediksi nilai melalui GAE; ε=0,2 ialah batas perubahan rasio [8, baris 131–139; 9, baris 179–258].')
p('L_total = L_policy + c_v L_value − c_e H adalah bentuk konseptual. c_v koefisien value loss, c_e koefisien entropi, H entropi aksi. Implementasi melakukan backward actor atas L_policy − c_e H dan backward critic terpisah atas c_v L_value; bukan satu tensor loss gabungan [8, baris 146–160].')
h('3.5 Pemetaan paper terhadap kode',2)
table(['Komponen','Implementasi lokal'],[['Observasi dan reward','vendor/on-policy/onpolicy/envs/mpe/scenarios/simple_spread.py'],['Actor/critic','vendor/on-policy/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py'],['Rollout','vendor/on-policy/onpolicy/runner/shared/mpe_runner.py'],['GAE','vendor/on-policy/onpolicy/utils/shared_buffer.py'],['PPO clipping/loss','vendor/on-policy/onpolicy/algorithms/r_mappo/r_mappo.py'],['Noise/test','tools/eval_core.py; tools/evaluate_noise.py']])
h('4. Formulasi Model dan Simulasi')
h('4.1 Agen, landmark, observasi, aksi, reward',2); p('Lingkungan memiliki 3 agen dan 3 landmark. Actor memilih aksi diskret dari observasi setiap agen. Reward episode merupakan penjumlahan reward per langkah dari agen pertama; dunia bersifat kolaboratif sehingga reward sama pada agen [3,4].')
h('4.2 Asumsi',2); p('Checkpoint, test seed, horizon, dan kebijakan aksi deterministik dipertahankan antar kondisi. Perubahan yang disengaja hanya simpangan baku noise observasi fisik.')
h('4.3 Formulasi noise',2); p("o′_t = o_t + ε_t, ε_t ~ N(0, σ²). Fitur komunikasi tidak diganggu. Generator noise diinisialisasi dari episode seed dan tingkat σ; σ ialah simpangan baku, bukan varians [4].")
h('4.4 Input dan output',2); p('Input: checkpoint actor terpilih, seed test 2001–2100, σ, dan lingkungan. Output: reward episode, success, collision pairs, jumlah jarak minimum landmark pada langkah akhir, jumlah landmark terisi, dan ringkasan CSV [4,10].')
h('5. Metodologi DEGREE')
for sub,body in [
('5.1 Define','Masalah: ketahanan koordinasi R-MAPPO saat observasi lokal fisik terganggu noise.'),
('5.2 Establish','Metrik: reward, success rate, collision pairs, final landmark distance, waktu training, serta variasi antarseed. Success didefinisikan sebagai semua tiga landmark terisi pada langkah akhir.'),
('5.3 Generate','Alternatif adalah skenario evaluasi σ=0; 0,05; 0,10; 0,20; 0,30, bukan model yang dilatih ulang.'),
('5.4 Rank','Reward dan success lebih tinggi lebih baik; collision dan final distance lebih rendah lebih baik. Pemeringkatan dilakukan per metrik karena tidak ada bobot gabungan yang ditetapkan.'),
('5.5 Evaluate and Iterate','Script menyediakan smoke test, pilot, training tiga baseline, pemilihan checkpoint, serta audit kelengkapan. Log smoke berakhir dengan interupsi dan metadata pilot tidak ditemukan; kedua tahap tidak boleh diklaim sukses hanya dari rencana script.'),
('5.6 Execute and Evaluate','Analisis akhir menggunakan 1.500 episode test lengkap. Reward rata-rata paling rendah pada σ=0,30, tetapi variasi antar seed dan success nol membatasi keputusan ketahanan.')]:
    h(sub,2); p(body)
h('6. Reproduksi Terbatas')
h('6.1 Lingkungan eksperimen',2); p('Log run mencatat MPE simple_spread dan R-MAPPO pada Windows. Metadata sistem mencatat CPU AMD64 16 logical cores dan NVIDIA RTX A1000 6 GB, tetapi `torch` tidak terdeteksi di metadata serta CUDA tercatat tidak tersedia. Penggunaan GPU oleh training tidak dapat dipastikan dari metadata itu [11,12].')
h('6.2 Konfigurasi',2); p('Protocol dan perintah baseline mencatat 2.000.000 environment steps per seed, 8 rollout threads, episode length 25, 10 PPO epoch, 1 mini batch, learning rate actor/critic 0,0007, γ=0,99, λ_GAE=0,95, dan clip 0,2 [11,12].')
h('6.3 Penyimpangan dari paper',2)
table(['Aspek','Paper/repository','Lokal'],[['Cakupan','Empat testbed pada paper [1]','Satu skenario MPE Spread'],['Environment steps','Script upstream Spread 20 juta [13]','2 juta per baseline'],['Training seed MPE','10 pada paper [1]','3'],['Rollout threads','128 pada script upstream [13]','8'],['Episode length','25 pada script upstream [13]','25'],['PPO epoch','10 pada script upstream [13]','10'],['Learning rate','0,0007 pada script upstream [13]','0,0007'],['Pembanding','MAPPO, IPPO, QMix, MADDPG [1]','R-MAPPO saja'],['Perangkat keras','Desktop 64-core CPU, RTX 3090 [1]','Metadata laptop AMD64; RTX A1000 terdeteksi'],['Python/library','Tidak disamakan langsung','Metadata Python 3.13.2, torch null; README 3.10']])
h('6.4 Smoke test dan pilot',2); p('Script smoke menargetkan 10.000 steps dan pilot 100.000 steps. Berkas smoke ada, tetapi log smoke berakhir dengan KeyboardInterrupt; metadata pilot tidak ditemukan. Karena itu status penyelesaian kedua tahap tidak terverifikasi [14].')
h('6.5 Training tiga seed',2); p('Tiga metadata baseline masing-masing mencatat return_code 0, 2 juta steps, seed 1–3, dan waktu training 3.944,5; 5.025,9; 4.389,0 detik. Learning curve tersedia, tetapi grafik bukan pengganti CSV evaluasi [12,15].')
h('6.6 Pemilihan checkpoint',2); p('Setiap kandidat dievaluasi pada 20 seed validasi 1001–1020 dengan noise nol. Peringkat menggunakan mean reward lalu success rate; snapshot teratas disalin identik ke outputs/best [16].')
table(['Seed','Snapshot','Kandidat','Reward validasi'],[[v['seed'],v['selected'],v['candidates'],fmt(v['mean'])] for v in validation])
h('7. Eksplorasi Noise Observasi')
h('7.1 Desain eksperimen',2); p('Baseline dilatih tiga kali pada observasi bersih. Noise hanya saat evaluasi; tidak ada training ulang per kondisi. Checkpoint actor yang sama untuk setiap seed dipakai pada kelima tingkat noise [4,10].')
h('7.2 Variabel bebas, terikat, dan kontrol',2); p('Variabel bebas: σ. Variabel terikat: reward, success, collision, final distance. Kontrol: checkpoint, seed test, horizon 25, skenario, dan aksi deterministik.')
h('7.3 Protokol test',2); p('Setiap kombinasi 3 training seed × 5 σ diuji pada 100 episode berseed 2001–2100. Semua 15 sel lengkap dan kunci episode unik. Test dipisahkan dari validasi [4,10].')
h('8. Hasil')
h('8.1 Learning curve baseline',2); fig('baseline_learning_curves.png','Gambar 1. Kurva pembelajaran baseline tiga seed. Grafik menunjukkan proses training; perbandingan noise memakai data test CSV.')
h('8.2 Tabel hasil setiap training seed',2)
table(['Seed','σ','Reward','Success','Collision','Jarak akhir'],[[int(r.train_seed),fmt(r.noise_std,2),fmt(r.mean_reward),fmt(100*r.success_rate,1)+'%',fmt(r.mean_collisions),fmt(r.mean_final_distance,3)] for _,r in PER.iterrows()])
h('8.3 Ringkasan antarseed',2)
table(['σ','Reward mean ± SD','Success','Collision mean ± SD','Jarak mean ± SD','Degradasi reward'],[[fmt(r.noise_std,2),fmt(r.reward_mean)+' ± '+fmt(r.reward_std),fmt(100*r.success_mean,1)+'%',fmt(r.collisions_mean)+' ± '+fmt(r.collisions_std),fmt(r.distance_mean,3)+' ± '+fmt(r.distance_std,3),fmt(r.reward_degradation_pct,2)+'%'] for _,r in main.iterrows()])
p('SD adalah simpangan baku sampel dari tiga rerata training seed, bukan variasi seluruh 300 episode sebagai unit independen. Degradasi dihitung (reward baseline − reward kondisi)/|reward baseline| × 100; nilai negatif berarti perbaikan numerik.')
h('8.4 Reward terhadap noise',2); fig('noise_vs_reward.png','Gambar 2. Rerata reward test dan SD antarseed menurut σ.')
h('8.5 Success rate terhadap noise',2); fig('noise_vs_success.png','Gambar 3. Success rate nol di semua kondisi; grafik tidak dapat memisahkan tingkat noise.')
h('8.6 Collision terhadap noise',2); fig('noise_vs_collisions.png','Gambar 4. Collision pairs per episode; penurunan pada noise tinggi tidak berarti reward membaik.')
h('9. Pembahasan')
p(f'Perubahan terhadap σ=0: pada σ=0,05 reward {fmt(main.iloc[1].delta_reward_mean)} (membaik), collision {fmt(main.iloc[1].delta_collisions_mean)}, dan jarak akhir +{fmt(main.iloc[1].delta_distance_mean,3)}; pada σ=0,10 reward {fmt(main.iloc[2].delta_reward_mean)}, collision +{fmt(main.iloc[2].delta_collisions_mean)}, jarak +{fmt(main.iloc[2].delta_distance_mean,3)}; pada σ=0,20 reward {fmt(main.iloc[3].delta_reward_mean)}, collision {fmt(main.iloc[3].delta_collisions_mean)}, jarak +{fmt(main.iloc[3].delta_distance_mean,3)}; pada σ=0,30 reward {fmt(main.iloc[4].delta_reward_mean)}, collision {fmt(main.iloc[4].delta_collisions_mean)}, jarak +{fmt(main.iloc[4].delta_distance_mean,3)}. Success tetap nol pada semua kondisi.')
p('Penurunan reward terbesar terhadap baseline terjadi pada σ=0,30. Tren reward rerata tidak monoton dari 0 ke 0,05; pola tiap seed juga berbeda: seed 1 dan 3 membaik pada 0,05, seed 2 memburuk, dan seed 3 pada 0,30 lebih baik daripada baseline seed itu. Collision juga tidak berhubungan sederhana dengan reward karena reward mengandung jarak landmark dan penalti self-collision, sementara metrik collision mengecualikan self-collision. Dugaan penyebab perubahan kecil mencakup variasi episode dan respons aksi terhadap perturbasi; desain ini tidak mengisolasi mekanisme tersebut.')
p('Peringkat reward (terbaik→terburuk): σ=0,05; 0; 0,10; 0,20; 0,30. Peringkat collision (paling sedikit→terbanyak): 0,30; 0,20; 0,05; 0; 0,10. Peringkat jarak akhir (paling dekat→terjauh): 0; 0,10; 0,05; 0,20; 0,30. Success seluruhnya seri. Tidak ada satu peringkat gabungan yang dapat dipertanggungjawabkan tanpa bobot metrik.')
h('10. Keterbatasan')
p('Reproduksi terbatas hanya mencakup satu skenario, tiga training seed, dan 2 juta environment steps, dibanding 20 juta pada script Spread upstream. Success rate mengalami floor effect pada definisi akhir episode. Metadata Python/PyTorch tidak konsisten; hasil smoke dan pilot tidak terverifikasi selesai. Tidak ada uji statistik atau interval ketidakpastian inferensial. Hasil tidak mendukung klaim signifikansi, generalisasi, maupun hubungan monoton menyeluruh.')
h('11. Kesimpulan')
p(f'Dalam pengujian lokal, rerata reward pada σ=0,30 lebih rendah {fmt(main.iloc[-1].reward_degradation_pct)}% daripada kondisi bersih dan jarak landmark akhir lebih besar. Namun perbaikan numerik pada σ=0,05, perbedaan antar seed, dan success nol menunjukkan ketahanan tidak dapat dinilai dengan satu pola monoton atau satu metrik. Kesimpulan berlaku hanya untuk checkpoint, seed, dan skenario yang diaudit.')
h('DAFTAR PUSTAKA')
refs=[
'[1] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS. references/NeurIPS-2022-the-surprising-effectiveness-of-ppo-in-cooperative-multi-agent-games-Paper-Datasets_and_Benchmarks.pdf.',
'[2] Bettini, M., Prorok, A., & Moens, V. (2024). BenchMARL: Benchmarking Multi-Agent Reinforcement Learning. Journal of Machine Learning Research, 25, 1–10. references/23-1612.pdf.',
'[3] vendor/on-policy/onpolicy/envs/mpe/scenarios/simple_spread.py, fungsi reward, observation, info.',
'[4] tools/eval_core.py, fungsi evaluate_episode.',
'[5] vendor/on-policy/onpolicy/runner/shared/mpe_runner.py.',
'[6] CODE_MAP.md.',
'[7] vendor/on-policy/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py.',
'[8] vendor/on-policy/onpolicy/algorithms/r_mappo/r_mappo.py.',
'[9] vendor/on-policy/onpolicy/utils/shared_buffer.py.',
'[10] tools/evaluate_noise.py; outputs/evaluation/noise_episode_results.csv; noise_results_per_training_seed.csv; noise_summary.csv.',
'[11] config/protocol.json; outputs/metadata/system_info.json; README.md.',
'[12] outputs/runs/baseline_seed1.json; baseline_seed2.json; baseline_seed3.json.',
'[13] vendor/on-policy/onpolicy/scripts/train_mpe_scripts/train_mpe_spread.sh.',
'[14] scripts/01_smoke_test.sh; scripts/02_pilot.sh; outputs/logs/smoke_seed1.log; outputs/runs/smoke_seed1.json.',
'[15] outputs/figures/baseline_learning_curves.png.',
'[16] tools/select_checkpoint.py; outputs/validation/seed_1_episodes.csv sampai seed_3_summary.csv; outputs/best/seed_1/selection.json sampai seed_3/selection.json.'
]
for x in refs:p(x)
h('LAMPIRAN REPRODUKSIBILITAS')
p('Berkas audit terpisah: reports/Ringkasan_Audit_Eksperimen.md. Tabel data turunan: reports/Tabel_Hasil_Utama.csv. Perhitungan ulang memeriksa jumlah baris, kunci unik, nilai hingga, seed test/validasi, dan kesesuaian agregat. Hasil asli pada outputs tidak diubah.')
p('Persamaan degradasi: D(σ) = [R(0) − R(σ)] / |R(0)| × 100%. Nilai negatif pada σ=0,05 menandai perbaikan reward numerik kecil. Seluruh SD yang ditampilkan bersifat deskriptif antarseed.')
p('Jejak program pembangun laporan: tools/build_report_artifacts.py. Eksekusi script hanya membaca data sumber dan menulis berkas dalam reports/.')
doc.save(OUT/'Laporan_MAPPO_Noise_DEGREE.docx')
print('CREATED', *(str(x) for x in OUT.iterdir()),sep='\n')
