import numpy as np
import skfuzzy
import matplotlib.pyplot as plt

counter = 0

for input_nilai_vokal in range(0,10):
    for input_nilai_teknik in range(0,10):
        for input_nilai_pembawaan in range(0,10):
            for input_nilai_penampilan in range(0,10):
                counter = counter+1
                print(counter, ":")

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

                # Komposisi Rule
                agregat = np.fmax(aktivasi_rule1, np.fmax(aktivasi_rule2, np.fmax(aktivasi_rule3, np.fmax(aktivasi_rule4, np.fmax(aktivasi_rule5, aktivasi_rule6)))))

                # Defuzzifikasi
                rating_akhir = skfuzzy.defuzz(rating, agregat, 'centroid')
                print("Rating Akhir :", rating_akhir)