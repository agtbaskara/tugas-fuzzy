import numpy as np
import skfuzzy
import matplotlib.pyplot as plt

# Pemodelan Fungsi Anggota Fuzzy
vokal = np.arange(0, 11, 1)
teknik = np.arange(0, 11, 1)
pembawaan = np.arange(0, 11, 1)
penampilan = np.arange(0, 11, 1)
rating = np.arange(0, 11, 1)

vokal_jelek = skfuzzy.trimf(vokal, [0, 0, 5])
vokal_biasa = skfuzzy.trapmf(vokal, [2, 3, 6, 8])
vokal_bagus = skfuzzy.trimf(vokal, [7, 10, 10])

teknik_jelek = skfuzzy.trimf(teknik, [0, 0, 5])
teknik_biasa = skfuzzy.trapmf(teknik, [2, 3, 6, 8])
teknik_bagus = skfuzzy.trimf(teknik, [6, 10, 10])

pembawaan_jelek = skfuzzy.trimf(pembawaan, [0, 0, 4])
pembawaan_biasa = skfuzzy.trapmf(pembawaan, [2, 3, 6, 9])
pembawaan_bagus = skfuzzy.trimf(pembawaan, [5, 10, 10])

penampilan_jelek = skfuzzy.trimf(penampilan, [0, 0, 5])
penampilan_biasa = skfuzzy.trapmf(penampilan, [3, 4, 7, 10])
penampilan_bagus = skfuzzy.trimf(penampilan, [8, 10, 10])

rating_jelek = skfuzzy.trimf(penampilan, [0, 0, 4])
rating_biasa = skfuzzy.trapmf(penampilan, [2, 4, 6, 8])
rating_bagus = skfuzzy.trimf(penampilan, [6, 10, 10])

# Plot Fungsi Keanggotaan
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(8, 9))

ax1.plot(vokal, vokal_jelek, 'b', linewidth=1.5, label='Jelek')
ax1.plot(vokal, vokal_biasa, 'g', linewidth=1.5, label='Biasa')
ax1.plot(vokal, vokal_bagus, 'r', linewidth=1.5, label='Bagus')
ax1.set_title('Fungsi Keanggotaan Vokal')
ax1.legend()

ax2.plot(teknik, teknik_jelek, 'b', linewidth=1.5, label='Jelek')
ax2.plot(teknik, teknik_biasa, 'g', linewidth=1.5, label='Biasa')
ax2.plot(teknik, teknik_bagus, 'r', linewidth=1.5, label='Bagus')
ax2.set_title('Fungsi Keanggotaan Teknik')
ax2.legend()

ax3.plot(pembawaan, pembawaan_jelek, 'b', linewidth=1.5, label='Jelek')
ax3.plot(pembawaan, pembawaan_biasa, 'g', linewidth=1.5, label='Biasa')
ax3.plot(pembawaan, pembawaan_bagus, 'r', linewidth=1.5, label='Bagus')
ax3.set_title('Fungsi Keanggotaan Pembawaan')
ax3.legend()

ax4.plot(penampilan, penampilan_jelek, 'b', linewidth=1.5, label='Jelek')
ax4.plot(penampilan, penampilan_biasa, 'g', linewidth=1.5, label='Biasa')
ax4.plot(penampilan, penampilan_bagus, 'r', linewidth=1.5, label='Bagus')
ax4.set_title('Fungsi Keanggotaan Penampilan')
ax4.legend()

# Input Nilai
input_nilai_vokal = float(input("Nilai Vokal : "))
input_nilai_teknik = float(input("Nilai Teknik : "))
input_nilai_pembawaan = float(input("Nilai Pembawaan : "))
input_nilai_penampilan = float(input("Nilai Penampilan : "))

vokal_nilai_jelek = skfuzzy.interp_membership(vokal, vokal_jelek, input_nilai_vokal)
vokal_nilai_biasa = skfuzzy.interp_membership(vokal, vokal_biasa, input_nilai_vokal)
vokal_nilai_bagus = skfuzzy.interp_membership(vokal, vokal_bagus, input_nilai_vokal)

teknik_nilai_jelek = skfuzzy.interp_membership(teknik, teknik_jelek, input_nilai_teknik)
teknik_nilai_biasa = skfuzzy.interp_membership(teknik, teknik_biasa, input_nilai_teknik)
teknik_nilai_bagus = skfuzzy.interp_membership(teknik, teknik_bagus, input_nilai_teknik)

pembawaan_nilai_jelek = skfuzzy.interp_membership(pembawaan, pembawaan_jelek, input_nilai_pembawaan)
pembawaan_nilai_biasa = skfuzzy.interp_membership(pembawaan, pembawaan_biasa, input_nilai_pembawaan)
pembawaan_nilai_bagus = skfuzzy.interp_membership(pembawaan, pembawaan_bagus, input_nilai_pembawaan)

penampilan_nilai_jelek = skfuzzy.interp_membership(penampilan, penampilan_jelek, input_nilai_penampilan)
penampilan_nilai_biasa = skfuzzy.interp_membership(penampilan, penampilan_biasa, input_nilai_penampilan)
penampilan_nilai_bagus = skfuzzy.interp_membership(penampilan, penampilan_bagus, input_nilai_penampilan)

#[R1] IF vokal JELEK then rating JELEK
rule1 = vokal_nilai_jelek
aktivasi_rule1 = np.fmin(rule1, rating_jelek)

#[R2] IF penampilan JELEK then rating JELEK
rule2 = penampilan_nilai_jelek
aktivasi_rule2 = np.fmin(rule2, rating_jelek)

#[R3] IF vokal BIASA AND penampilan BAGUS then rating BAGUS
rule3 = np.fmin(vokal_nilai_biasa, penampilan_nilai_bagus)
aktivasi_rule3 = np.fmin(rule3, rating_bagus)

#[R4] IF vokal BAGUS OR teknik BAGUS then rating BAGUS
rule4 = np.fmax(vokal_nilai_bagus, teknik_nilai_bagus)
aktivasi_rule4 = np.fmin(rule4, rating_bagus)

#[R5] IF vokal BIASA AND pembawaan BAGUS then rating BAGUS
rule5 = np.fmin(vokal_nilai_biasa, pembawaan_nilai_bagus)
aktivasi_rule5 = np.fmin(rule5, rating_bagus)

#[R6] IF (vokal BIASA OR teknik BIASA) AND penampilan BIASA then rating BIASA
rule6 = np.fmin(penampilan_nilai_biasa, np.fmax(vokal_nilai_biasa, teknik_nilai_biasa))
aktivasi_rule6 = np.fmin(rule6, rating_biasa)

# Plot Output Rule
rating0 = np.zeros_like(rating)
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(rating, rating0, aktivasi_rule1, facecolor='r', alpha=0.7)
ax0.plot(rating, rating_jelek, 'r', linewidth=0.5, linestyle='--', )
ax0.fill_between(rating, rating0, aktivasi_rule2, facecolor='b', alpha=0.7)
ax0.plot(rating, rating_jelek, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(rating, rating0, aktivasi_rule3, facecolor='g', alpha=0.7)
ax0.plot(rating, rating_bagus, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(rating, rating0, aktivasi_rule4, facecolor='c', alpha=0.7)
ax0.plot(rating, rating_bagus, 'c', linewidth=0.5, linestyle='--', )
ax0.fill_between(rating, rating0, aktivasi_rule5, facecolor='m', alpha=0.7)
ax0.plot(rating, rating_bagus, 'm', linewidth=0.5, linestyle='--', )
ax0.fill_between(rating, rating0, aktivasi_rule6, facecolor='y', alpha=0.7)
ax0.plot(rating, rating_biasa, 'y', linewidth=0.5, linestyle='--', )
ax0.set_title('Output Rule')

# Komposisi Rule
agregat = np.fmax(aktivasi_rule1, np.fmax(aktivasi_rule2, np.fmax(aktivasi_rule3, np.fmax(aktivasi_rule4, np.fmax(aktivasi_rule5, aktivasi_rule6)))))

# Defuzzifikasi
rating_akhir = skfuzzy.defuzz(rating, agregat, 'centroid')
print("Rating Akhir :", rating_akhir)

# Plot Komposisi Rule
rating_aktivasi = skfuzzy.interp_membership(rating, agregat, rating_akhir)
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(rating, rating_jelek, 'r')
ax0.plot(rating, rating_biasa, 'g')
ax0.plot(rating, rating_bagus, 'b')
ax0.fill_between(rating, rating0, aktivasi_rule1, facecolor='gray')
ax0.fill_between(rating, rating0, aktivasi_rule2, facecolor='gray')
ax0.fill_between(rating, rating0, aktivasi_rule3, facecolor='gray')
ax0.fill_between(rating, rating0, aktivasi_rule4, facecolor='gray')
ax0.fill_between(rating, rating0, aktivasi_rule5, facecolor='gray')
ax0.fill_between(rating, rating0, aktivasi_rule6, facecolor='gray')
ax0.plot([rating_akhir, rating_akhir], [0, rating_aktivasi], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Komposisi Rule')

plt.show()