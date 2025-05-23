# predictive_analytics

# Laporan Proyek Machine Learning - Damianus Christopher Samosir

## Domain Proyek

Diabetes Mellitus merupakan salah satu penyakit kronis yang memengaruhi jutaan orang di seluruh dunia, termasuk populasi spesifik seperti Suku Pima Indian yang memiliki prevalensi tinggi terhadap penyakit ini. Penyakit ini ditandai dengan kadar glukosa darah yang tinggi, yang dapat menyebabkan komplikasi serius jika tidak dideteksi dan dikelola dengan baik. Deteksi dini diabetes sangat penting untuk mencegah komplikasi jangka panjang seperti penyakit kardiovaskular, kerusakan ginjal, dan gangguan penglihatan. Dengan memanfaatkan teknik *machine learning*, kita dapat membangun model prediktif untuk mengidentifikasi individu yang berisiko tinggi terkena diabetes berdasarkan data klinis.

Proyek ini bertujuan untuk mengembangkan model *machine learning* yang dapat memprediksi risiko diabetes pada populasi Suku Pima Indian menggunakan dataset Pima Indians Diabetes. Masalah ini perlu diselesaikan karena deteksi dini dapat membantu dalam intervensi medis yang tepat waktu, mengurangi biaya perawatan kesehatan, dan meningkatkan kualitas hidup pasien. Penelitian sebelumnya, seperti yang dilakukan oleh Smith et al. [1], menunjukkan bahwa algoritma *machine learning* seperti Random Forest dan Decision Tree dapat memberikan akurasi yang baik dalam memprediksi diabetes berdasarkan fitur klinis.

**Referensi:**
[1] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," in *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 1988, pp. 261–265. [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/

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

Dataset yang digunakan dalam proyek ini adalah **Pima Indians Diabetes Dataset**, yang diunduh dari Google Drive dengan ID file `1rz1kAkPon56cdgse59oTkumkaD0N4J4h`. Dataset ini tersedia dalam format CSV (`diabetes.csv`) dan berisi 768 baris data dengan 9 kolom, termasuk 8 fitur independen dan 1 kolom target (*Outcome*). Dataset ini berasal dari studi tentang diabetes pada populasi Suku Pima Indian, yang memiliki prevalensi tinggi terhadap diabetes.

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

2. **Normalisasi Data**:
   - Data dinormalisasi menggunakan *StandardScaler* dari *scikit-learn* untuk menstandarisasi fitur ke skala yang sama.
   - **Alasan**: Normalisasi penting untuk algoritma seperti Naive Bayes yang sensitif terhadap skala data, meskipun Random Forest dan Decision Tree kurang sensitif.

3. **Pemisahan Data**:
   - Dataset dibagi menjadi data pelatihan (*X_train*, *y_train*) dan data pengujian (*X_test*, *y_test*) menggunakan *train_test_split* dengan rasio 80:20.
   - **Alasan**: Pemisahan data memungkinkan evaluasi model pada data yang tidak terlihat untuk menghindari *overfitting*.

## Modeling
Tiga algoritma *machine learning* digunakan untuk membangun model prediktif: Random Forest, Naive Bayes, dan J48 Decision Tree. Berikut adalah penjelasan tahapan dan parameter yang digunakan:

1. **Random Forest**:
   - **Parameter**: Model dioptimalkan menggunakan *GridSearchCV* untuk menentukan parameter terbaik (misalnya, jumlah pohon, kedalaman pohon). Parameter spesifik tidak ditampilkan dalam kode, tetapi model terbaik (*best_model*) digunakan untuk prediksi.
   - **Kelebihan**: Tahan terhadap *overfitting*, dapat menangani data tidak seimbang, dan memberikan pentingnya fitur.
   - **Kekurangan**: Kompleksitas komputasi tinggi untuk dataset besar, dan interpretasi model lebih sulit dibandingkan Decision Tree.
   - **Improvement**: *Hyperparameter tuning* dilakukan untuk menemukan kombinasi parameter terbaik, seperti jumlah pohon (*n_estimators*) dan kedalaman maksimum (*max_depth*), untuk meningkatkan akurasi dan *F1-score*.

2. **Naive Bayes**:
   - **Parameter**: Menggunakan *GaussianNB* tanpa *hyperparameter tuning* karena algoritma ini memiliki sedikit parameter yang dapat diatur.
   - **Kelebihan**: Cepat, sederhana, dan efektif untuk dataset dengan fitur independen.
   - **Kekurangan**: Asumsi independensi antar fitur sering kali tidak realistis, yang dapat menurunkan performa pada dataset dengan korelasi antar fitur.

3. **J48 Decision Tree**:
   - **Parameter**: Menggunakan *DecisionTreeClassifier* dengan kriteria *entropy* (setara dengan J48 di WEKA).
   - **Kelebihan**: Mudah diinterpretasikan, cepat untuk dataset kecil, dan dapat menangani data non-numerik.
   - **Kekurangan**: Rentan terhadap *overfitting* tanpa pengaturan parameter seperti kedalaman pohon atau jumlah sampel minimum.

### Pemilihan Model Terbaik
Random Forest dipilih sebagai model terbaik karena memiliki akurasi tertinggi (85%) dan *F1-score* yang seimbang (77.7%) dibandingkan Naive Bayes dan J48 Decision Tree. Selain itu, Random Forest lebih tahan terhadap *overfitting* dan memberikan hasil yang lebih stabil melalui pendekatan ensemble.

## Evaluation
Model dievaluasi menggunakan metrik **akurasi**, **precision**, **recall**, dan **F1-score**, yang sesuai untuk masalah klasifikasi biner seperti prediksi diabetes. Selain itu, kurva ROC dan AUC (*Area Under the Curve*) digunakan untuk mengevaluasi kemampuan model dalam membedakan kelas positif dan negatif.

### Penjelasan Metrik
1. **Akurasi**: Proporsi prediksi yang benar dari total prediksi.
   - Formula:  
     \[
     \text{Akurasi} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
     \]
     di mana TP = *True Positive*, TN = *True Negative*, FP = *False Positive*, FN = *False Negative*.
2. **Precision**: Proporsi prediksi positif yang benar dari semua prediksi positif.
   - Formula:  
     \[
     \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
     \]
3. **Recall**: Proporsi kasus positif yang berhasil diidentifikasi.
   - Formula:  
     \[
     \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
     \]
4. **F1-score**: Rata-rata harmonik dari *precision* dan *recall*, memberikan keseimbangan antara keduanya.
   - Formula:  
     \[
     \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
     \]
5. **AUC**: Mengukur area di bawah kurva ROC, yang menunjukkan kemampuan model untuk membedakan kelas positif dan negatif. Nilai AUC mendekati 1 menunjukkan model yang baik.

### Hasil Evaluasi
Berikut adalah hasil evaluasi dari ketiga model berdasarkan metrik yang digunakan:

| Model              | Akurasi | Precision | Recall | F1-Score |
|--------------------|---------|-----------|--------|----------|
| Random Forest      | 85%     | 79.22%    | 76.25% | 77.7%    |
| Naive Bayes        | 74%     | 65.22%    | 56.25% | 60.4%    |
| J48 Decision Tree  | 84%     | 78.94%    | 75%    | 76.9%    |

- **Random Forest**: Memberikan akurasi tertinggi (85%) dan *F1-score* yang seimbang (77.7%), menunjukkan kemampuan yang baik dalam memprediksi diabetes. Kurva ROC (ditampilkan dalam kode) menunjukkan AUC yang tinggi, mengindikasikan performa yang kuat dalam membedakan kelas.
- **Naive Bayes**: Memiliki akurasi terendah (74%) dan *recall* yang rendah (56.25%), menunjukkan bahwa model ini kurang efektif dalam mengidentifikasi kasus positif (diabetes).
- **J48 Decision Tree**: Menunjukkan akurasi yang baik (84%), tetapi *recall* sedikit lebih rendah dibandingkan Random Forest, yang dapat memengaruhi kemampuan mendeteksi kasus positif.

### Perbandingan dengan Artikel
Dibandingkan dengan artikel referensi, Random Forest pada proyek ini memiliki akurasi lebih tinggi (85% vs. 79.57%), tetapi *precision* dan *recall* lebih rendah. Naive Bayes pada artikel memiliki performa lebih baik dalam hal *recall* (86.75% vs. 56.25%), sementara J48 Decision Tree pada proyek lebih unggul dalam akurasi (84% vs. 74.78%).

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
Kurva ini menunjukkan bahwa Random Forest memiliki AUC tertinggi, diikuti oleh J48 Decision Tree, dan Naive Bayes memiliki AUC terendah.

## Kesimpulan
Proyek ini berhasil membangun model *machine learning* untuk memprediksi risiko diabetes pada populasi Suku Pima Indian dengan akurasi yang baik. Random Forest terbukti menjadi model terbaik dengan akurasi 85% dan *F1-score* 77.7%, diikuti oleh J48 Decision Tree (84%) dan Naive Bayes (74%). Teknik penggantian nilai nol dengan median dan normalisasi data meningkatkan kualitas dataset, terutama untuk Naive Bayes. Meskipun performa Random Forest lebih baik dibandingkan artikel referensi dalam hal akurasi, *recall* dan *precision* dapat ditingkatkan lebih lanjut dengan eksplorasi *hyperparameter tuning* yang lebih mendalam atau penggunaan teknik seperti *oversampling* untuk menangani ketidakseimbangan kelas.

