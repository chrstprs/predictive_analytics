# predictive_analytics

# Laporan Proyek Machine Learning - Damianus Christopher Samosir

## Domain Proyek: Kesehatan
### Latar Belakang

Diabetes Mellitus adalah salah satu penyakit kronis yang terus mengalami peningkatan prevalensi secara global. Penyakit ini terjadi akibat gangguan metabolisme glukosa yang menyebabkan kadar gula darah tinggi dalam jangka panjang, dan dapat menimbulkan komplikasi serius seperti penyakit jantung, gagal ginjal, neuropati, serta gangguan penglihatan. Salah satu populasi yang diketahui memiliki tingkat kejadian diabetes yang tinggi adalah Suku Pima Indian, yang menjadikannya subjek penting dalam penelitian epidemiologi dan prediksi diabetes.

Dengan kemajuan teknologi, pendekatan machine learning semakin banyak digunakan dalam bidang kesehatan untuk membangun model prediksi penyakit. Dataset Pima Indians Diabetes sering digunakan untuk mengembangkan model prediktif berdasarkan data klinis seperti kadar glukosa, tekanan darah, indeks massa tubuh, dan riwayat keluarga. Beberapa algoritma yang telah menunjukkan performa baik dalam studi sebelumnya meliputi Random Forest, Naive Bayes, dan J48 Decision Tree, dengan tingkat akurasi prediksi yang cukup tinggi.

Penggunaan machine learning dalam prediksi dini diabetes memiliki potensi besar untuk meningkatkan efektivitas intervensi medis dan mengurangi beban sistem kesehatan. Dengan identifikasi individu berisiko tinggi sejak awal, langkah-langkah pencegahan dapat dilakukan secara lebih tepat sasaran, efisien, dan personal.

Referensi:
- Kırğıl, E. N. H., Erkal, B., & Erçelebİ Ayyildiz, T. (2022). Predicting Diabetes Using Machine Learning Techniques. 2022 International Conference on Theoretical and Applied Computer Science and Engineering (ICTASCE), 137–141. https://doi.org/10.1109/ICTACSE50438.2022.10009726
- Singh, K., Rout, J. K., & Das, H. (2019). Diabetes Prediction using Machine Learning Techniques. 2019 International Conference on Intelligent Computing and Remote Sensing (ICICRS), 1–6. https://doi.org/10.1109/ICICRS46726.2019.955588

## Business Understanding

### Problem Statements
1. Bagaimana cara membangun model *machine learning* yang akurat untuk memprediksi risiko diabetes pada populasi Suku Pima Indian berdasarkan data klinis seperti kadar glukosa, tekanan darah, dan BMI?
2. Bagaimana performa berbagai algoritma *machine learning* (Random Forest, Naive Bayes, dan J48 Decision Tree) dalam memprediksi diabetes, dan algoritma mana yang memberikan hasil terbaik?
3. Bagaimana cara menangani nilai nol atau data yang hilang dalam dataset untuk meningkatkan kualitas model prediktif?

### Goals
1. Mengembangkan model *machine learning* yang memiliki akurasi tinggi (di atas 80%) untuk memprediksi risiko diabetes berdasarkan fitur klinis.
2. Membandingkan performa tiga algoritma *machine learning* (Random Forest, Naive Bayes, dan J48 Decision Tree) menggunakan metrik evaluasi seperti akurasi, *precision*, *recall*, dan *F1-score* untuk menentukan model terbaik.
3. Menerapkan teknik penggantian nilai nol menggunakan median untuk memastikan dataset berkualitas tinggi dan siap digunakan untuk pelatihan model.

### Solution Statements
1. Menggunakan tiga algoritma *machine learning*: Random Forest, Naive Bayes, dan J48 Decision Tree untuk membangun model prediktif, dengan evaluasi performa menggunakan metrik akurasi, *precision*, *recall*, dan *F1-score*.
2. Melakukan *hyperparameter tuning* pada model Random Forest untuk meningkatkan performa model dasar (*baseline model*).
3. Membandingkan performa ketiga algoritma berdasarkan metrik evaluasi dan memilih model dengan akurasi dan *F1-score* tertinggi sebagai solusi terbaik.

## Data Understanding

Proyek ini menggunakan dataset dari Kaggle berjudul Movie Recommendation Data yang dapat diunduh melalui tautan berikut: 🔗 https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/code, Dataset ini tersedia dalam format CSV (`diabetes.csv`) dan berisi 768 baris data dengan 9 kolom, termasuk 8 fitur independen dan 1 kolom target (*Outcome*). Dataset ini berasal dari studi tentang diabetes pada populasi Suku Pima Indian, yang memiliki prevalensi tinggi terhadap diabetes.

### Variabel-variabel pada Pima Indians Diabetes Dataset
- **Pregnancies**: Jumlah kehamilan yang dialami pasien (numerik).
- **Glucose**: Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral (mg/dL, numerik).
- **BloodPressure**: Tekanan darah diastolik (mm Hg, numerik).
- **SkinThickness**: Ketebalan lipatan kulit trisep (mm, numerik).
- **Insulin**: Kadar insulin serum 2 jam (mu U/ml, numerik).
- **BMI**: Indeks Massa Tubuh (berat dalam kg / (tinggi dalam m)^2, numerik).
- **DiabetesPedigreeFunction**: Fungsi silsilah diabetes, yang mengukur risiko genetik diabetes (numerik).
- **Age**: Usia pasien (tahun, numerik).
- **Outcome**: Variabel target yang menunjukkan apakah pasien memiliki diabetes (1 = positif, 0 = negatif, biner).

### Exploratory Data Analysis
- Dataset memiliki 768 baris dan 9 kolom, tanpa nilai *missing* eksplisit (NaN). Namun, terdapat nilai nol pada kolom seperti *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, dan *BMI*, yang tidak realistis secara klinis dan perlu ditangani.
- Distribusi *Outcome* menunjukkan bahwa sekitar 35% pasien memiliki diabetes (1) dan 65% tidak (0), menunjukkan ketidakseimbangan kelas yang ringan.
- Visualisasi data (meskipun tidak ditampilkan dalam kode yang diberikan) dapat dilakukan menggunakan *heatmap* korelasi untuk memahami hubungan antar fitur, atau histogram untuk melihat distribusi masing-masing fitur.

## Data Preparation
Proses persiapan data dilakukan untuk memastikan dataset bersih dan siap digunakan untuk pelatihan model. Berikut adalah tahapan yang dilakukan:

1. **Penggantian Nilai Nol**:
   - Nilai nol pada kolom *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, dan *BMI* digantikan dengan median dari masing-masing kolom berdasarkan kelompok *Outcome* (0 atau 1). 
   - **Alasan**: Nilai nol pada fitur-fitur ini tidak realistis secara klinis (misalnya, glukosa atau BMI tidak mungkin nol). Penggantian dengan median dipilih karena lebih tahan terhadap outlier dibandingkan rata-rata.
     ![image](https://github.com/user-attachments/assets/df83e725-a6ed-4290-9bb0-2e3bf8b903c8)
     ![image](https://github.com/user-attachments/assets/8cc6254d-7648-4b7e-bcf2-962daeb80db9)
     ![image](https://github.com/user-attachments/assets/bb61defb-bcb5-4c4d-abda-9277499e1114)
     ![image](https://github.com/user-attachments/assets/eceedd98-9af2-4121-a6ba-91060f4a310f)
     ![image](https://github.com/user-attachments/assets/8f3f534d-4e2d-4844-b3f7-f6461044d27f)

2. **Normalisasi Data**:
   - Data dinormalisasi menggunakan *StandardScaler* dari *scikit-learn* untuk menstandarisasi fitur ke skala yang sama.
   - **Alasan**: Normalisasi penting untuk algoritma seperti Naive Bayes yang sensitif terhadap skala data, meskipun Random Forest dan Decision Tree kurang sensitif.
     ![image](https://github.com/user-attachments/assets/67c56d0f-53f1-4fa4-8a35-edb8b2f9602a)

3. **Pemisahan Data**:
   - Dataset dibagi menjadi data pelatihan (*X_train*, *y_train*) dan data pengujian (*X_test*, *y_test*) menggunakan *train_test_split* dengan rasio 70:30.
   - **Alasan**: Pemisahan data memungkinkan evaluasi model pada data yang tidak terlihat untuk menghindari *overfitting*.
     ![image](https://github.com/user-attachments/assets/45f871e3-31ed-490b-a8d4-93c17c384a25)


## Modeling
Tiga algoritma *machine learning* digunakan untuk membangun model prediktif: Random Forest, Naive Bayes, dan J48 Decision Tree. Berikut adalah penjelasan tahapan dan parameter yang digunakan:

1. **Random Forest**:
   - **Parameter**: Model dioptimalkan menggunakan *GridSearchCV* untuk menentukan parameter terbaik (misalnya, jumlah pohon, kedalaman pohon). Parameter spesifik tidak ditampilkan dalam kode, tetapi model terbaik (*best_model*) digunakan untuk prediksi.
   - **Kelebihan**: Tahan terhadap *overfitting*, dapat menangani data tidak seimbang, dan memberikan pentingnya fitur.
   - **Kekurangan**: Kompleksitas komputasi tinggi untuk dataset besar, dan interpretasi model lebih sulit dibandingkan Decision Tree.
   - **Improvement**: *Hyperparameter tuning* dilakukan untuk menemukan kombinasi parameter terbaik, seperti jumlah pohon (*n_estimators*) dan kedalaman maksimum (*max_depth*), untuk meningkatkan akurasi dan *F1-score*.
     ![image](https://github.com/user-attachments/assets/cc502077-741a-4b10-868a-8821595ea7cf)


2. **Naive Bayes**:
   - **Parameter**: Menggunakan *GaussianNB* tanpa *hyperparameter tuning* karena algoritma ini memiliki sedikit parameter yang dapat diatur.
   - **Kelebihan**: Cepat, sederhana, dan efektif untuk dataset dengan fitur independen.
   - **Kekurangan**: Asumsi independensi antar fitur sering kali tidak realistis, yang dapat menurunkan performa pada dataset dengan korelasi antar fitur.
     ![image](https://github.com/user-attachments/assets/f2586cf3-6873-4332-8fef-996c0d894af8)


3. **J48 Decision Tree**:
   - **Parameter**: Menggunakan *DecisionTreeClassifier* dengan kriteria *entropy* (setara dengan J48 di WEKA).
   - **Kelebihan**: Mudah diinterpretasikan, cepat untuk dataset kecil, dan dapat menangani data non-numerik.
   - **Kekurangan**: Rentan terhadap *overfitting* tanpa pengaturan parameter seperti kedalaman pohon atau jumlah sampel minimum.
     ![image](https://github.com/user-attachments/assets/088791ec-4dd2-4c31-b861-68e5d258d1eb)


### Pemilihan Model Terbaik
Random Forest dipilih sebagai model terbaik karena memiliki akurasi tertinggi (85%) dan *F1-score* yang seimbang (77.7%) dibandingkan Naive Bayes dan J48 Decision Tree. Selain itu, Random Forest lebih tahan terhadap *overfitting* dan memberikan hasil yang lebih stabil melalui pendekatan ensemble.

## Evaluation
Model dievaluasi menggunakan metrik **akurasi**, **precision**, **recall**, dan **F1-score**, yang sesuai untuk masalah klasifikasi biner seperti prediksi diabetes. Selain itu, kurva ROC dan AUC (*Area Under the Curve*) digunakan untuk mengevaluasi kemampuan model dalam membedakan kelas positif dan negatif.

### Penjelasan Metrik
1. **Akurasi**: Proporsi prediksi yang benar dari total prediksi.
   - Formula:
     
   $$
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   $$

      di mana TP = *True Positive*, TN = *True Negative*, FP = *False Positive*, FN = *False Negative*.
2. **Precision**: Proporsi prediksi positif yang benar dari semua prediksi positif.
   - Formula:

$$
 \text{Precision} = \frac{TP}{TP + FP}
$$

3. **Recall**: Proporsi kasus positif yang berhasil diidentifikasi.
   - Formula:  

$$
 \text{Recall} = \frac{TP}{TP + FN}
$$

4. **F1-score**: Rata-rata harmonik dari *precision* dan *recall*, memberikan keseimbangan antara keduanya.
   - Formula:  
     
$$
 \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

5. **AUC**: Mengukur area di bawah kurva ROC, yang menunjukkan kemampuan model untuk membedakan kelas positif dan negatif. Nilai AUC mendekati 1 menunjukkan model yang baik.

### Hasil Evaluasi
Berikut adalah hasil evaluasi dari ketiga model berdasarkan metrik yang digunakan:

| Model              | Akurasi | Precision | Recall | F1-Score |
|--------------------|---------|-----------|--------|----------|
| Random Forest      | 85%     | 80.26%    | 76.25% | 78.2%    |
| Naive Bayes        | 74%     | 65.22%    | 56.25% | 60.4%    |
| J48 Decision Tree  | 85%     | 79.22%    | 76.25% | 77.7%    |

- **Random Forest**: Memberikan akurasi tertinggi (85%) dan *F1-score* yang seimbang (78.2%), menunjukkan kemampuan yang baik dalam memprediksi diabetes. Kurva ROC (ditampilkan dalam kode) menunjukkan AUC yang tinggi, mengindikasikan performa yang kuat dalam membedakan kelas.
- **Naive Bayes**: Memiliki akurasi terendah (74%) dan *recall* yang rendah (56.25%), menunjukkan bahwa model ini kurang efektif dalam mengidentifikasi kasus positif (diabetes).
- **J48 Decision Tree**: Menunjukkan akurasi yang baik (85%), tetapi *Precision* sedikit lebih rendah dibandingkan Random Forest, yang dapat memengaruhi kemampuan mendeteksi kasus positif.

### Visualisasi ROC Curve
Kurva ROC untuk ketiga model diplot untuk membandingkan performa:
```python
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_score_rf:.2f})", color='blue')
plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_score_nb:.2f})", color='red')
plt.plot(fpr_j48, tpr_j48, label=f"J48 Decision Tree (AUC = {auc_score_j48:.2f})", color='green')
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest, Naive Bayes, & J48 Decision Tree")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/614a2e1a-3311-4a18-a3d3-7bc01e50af49)


Kurva ini menunjukkan bahwa Random Forest memiliki AUC tertinggi, diikuti oleh J48 Decision Tree, dan Naive Bayes memiliki AUC terendah.

## Kesimpulan
Proyek ini berhasil membangun model *machine learning* untuk memprediksi risiko diabetes pada populasi Suku Pima Indian dengan akurasi yang baik. Random Forest terbukti menjadi model terbaik dengan akurasi 85% dan *F1-score* 78.20%, diikuti oleh J48 Decision Tree (85%) dan Naive Bayes (74%). Teknik penggantian nilai nol dengan median dan normalisasi data meningkatkan kualitas dataset, terutama untuk Naive Bayes. Meskipun performa Random Forest lebih baik dibandingkan artikel referensi dalam hal akurasi, *recall* dan *precision* dapat ditingkatkan lebih lanjut dengan eksplorasi *hyperparameter tuning* yang lebih mendalam atau penggunaan teknik seperti *oversampling* untuk menangani ketidakseimbangan kelas.

