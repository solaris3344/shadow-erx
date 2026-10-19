# Sonuç kaydı: Synthetic sahnesi (ön kayıtlı birincil analiz)

**Tarih:** 18.10.2026
**Ön kayıt:** `PREREG_TR.md`, SHA-256 `668169E7906C8137D686033EF6DD4F9049DF81FE35AB13F0A615D522252BFCE2`. Hash analizden önce alındı.
**Sapmalar:** Yok. Parametre değişikliği yapılmadı. İlk indirme denemesi bozuk dosya verdi; dosya yeniden indirildi (600 × 2400 × 90, float64), hiçbir analiz bozuk dosyayla tamamlanmadı.
**Çıktılar:** `results_synthetic_v13`, `results_synthetic_v12`

## 1. Birincil sonuçlar (v13, `common_scored`, line_z, 3 tohum ortalaması)

| Yöntem | AUROC | AP | FPR90 | ms/satır |
|---|---|---|---|---|
| shadow_cc | **0,8609** | **0,0479** | **0,3409** | 0,654 |
| shadow_nc | 0,8520 | 0,0460 | 0,3727 | 0,455 |
| delayed_trim10_cc | 0,8183 | 0,0409 | 0,4840 | 0,266 |
| delayed_trim05_cc | 0,7958 | 0,0366 | 0,5459 | 0,283 |
| official_erx | 0,7861 | 0,0366 | 0,5486 | 0,291 |
| official_rt_ck_rxd | 0,5821 | 0,0361 | 0,8806 | 10,089 |
| official_lbl_ad | 0,5731 | 0,0155 | 0,9968 | 0,605 |
| official_rx_bil | 0,5672 | 0,0397 | 0,8894 | 3,343 |
| frozen | 0,5591 | 0,0259 | 0,9987 | 0,077 |
| official_rx_baseline | 0,5327 | 0,0420 | 0,9140 | 36,784 (49 satır gecikmeli) |

ms/satır: tek iş parçacığı. Bizim yöntemlerde ortak projeksiyon süresi hariç, resmi ERX'te dahil.

## 2. Hipotez kararları

| Hipotez | Kural | Gözlem | Karar |
|---|---|---|---|
| H1: ERX'e göre iyileşme | 3 metrikte aynı yönde iyileşme | AUROC +0,075, AP +0,011, FPR90 −0,208 | **Destek** |
| H2: Sabit kırpmaya göre pişmanlık | Δ AP > +0,02 kazanç; ±0,02 pişmanlıksız | Δ = +0,0070 | **Pişmanlıksız** |
| H3: Zamanlama (koşullu) | S ≥ 0,02 ise ≥12/15 permütasyon | S = 0,0042 (v12 line_z) | **Bilgisiz** |
| H4: Konum (betimsel) | Sıralama raporlanır | 10 yöntem içinde AUROC, AP ve FPR90'da 1. | Betimsel |

## 3. Betimsel ve keşifsel bulgular (doğrulayıcı değildir)

**3.1 Permütasyon kontrolü (v12).** Shadow, aynı aksiyon sayılı karıştırılmış kopyaların AP'de 15/15'ini her iki modda geçti.

| Mod | shadow_cc | shadow_cc_permgate |
|---|---|---|
| line_z | AUROC 0,860, AP 0,047, FPR90 0,343 | AUROC 0,758, AP 0,041, FPR90 0,989 |
| raw | AUROC 0,917, AP 0,263, FPR90 0,201 | AUROC 0,888, AP 0,114, FPR90 0,201 |

Line_z modunda karıştırılmış kapı FPR90'da Frozen düzeyine (0,999) çöküyor, Shadow 0,343'te kalıyor. Bu, güncelleme oranı aynı tutulduğunda kabul edilen güncellemelerin zamanlamasının belirleyici olduğunu gösteriyor. Ancak H3 ön kayıtta S koşuluna bağlandığı için bu bulgu **keşifsel** olarak raporlanır.

**3.2 Ön kayıt tasarımına ilişkin ders.** H3'teki moderatör S (Trim05-Trim10 yayılımı), skip oranının yüksek olduğu bir sahneyi öngörmedi. Synthetic'te Shadow satırların %77-80'inde güncellemeyi atlıyor; asıl karşıtlık Trim05 ile Trim10 arasında değil, güncelleme ile atlama arasında. S, bu sahnede zamanlamanın değerini yakalamadı. Gelecek sahneler için daha uygun moderatör, Frozen ile en iyi sabit kırpma arasındaki fark olabilir. Bu öneri sonuçlar görüldükten sonra yapılmıştır ve yalnızca yeni bir ön kayıtta kullanılabilir.

**3.3 H2 eşiğinin sınırlılığı.** AP'nin mutlak değeri bu sahnede çok düşük (0,03-0,05). 0,02'lik mutlak eşik göreli olarak yaklaşık %50'ye karşılık geliyor. Gözlenen göreli fark +%17. Ayrıca AUROC'ta +0,043 ve FPR90'da −0,143 fark var. Bunlar betimseldir; H2'nin kararı değişmez.

**3.4 Tutarlılık düzeltmesi.** `shadow_cc`, `shadow_nc`'yi üç metrikte de geçti (v13 ve v12). Yön Beach ile aynı; ana yöntem olarak `cc` seçimini destekliyor.

**3.5 Frozen.** Synthetic'te Frozen çöküyor (line_z AUROC 0,559). Arka plan güncellemesi bu sahnede zorunlu; Shadow buna rağmen satırların yalnızca ~%20'sinde güncelleme kabul ediyor.

## 4. Yeniden üretim kontrolü

Resmi dedektörlerin `official_full` AUROC değerleri ERX makalesindeki Synthetic tablosuyla uyumlu: RX Baseline 0,531 (makale 0,531), RT-CK-RXD 0,581 (0,581), RX-BIL 0,566 (0,566 ± 0,002), LBL-AD 0,575 (0,576), ERX 0,787 (0,751 ± 0,032). Beach ve SNP'de de dört deterministik dedektör ±0,001 içinde eşleşmişti.

## 5. Üç sahnenin özeti (line_z, `common_scored`)

| Sahne | Statü | shadow_cc vs ERX | shadow_cc vs en iyi sabit kırpma (AP) | Permütasyon (AP) | 10 yöntem içinde |
|---|---|---|---|---|---|
| Beach | geliştirme | 3/3 iyileşme | +0,100 | 15/15 | 3 metrikte 1. |
| SNP | geliştirme | 3/3 iyileşme | −0,004 | 10/15 | orta sıralar |
| Synthetic | **ön kayıtlı** | 3/3 iyileşme (H1) | +0,007 (H2 pişmanlıksız) | 15/15 (keşifsel) | 3 metrikte 1. |

## 6. İkincil analiz: Synthetic ters yön (`--reverse`)

Ters yön, ön kayıtta ikincil kanıt olarak tanımlandı. Aynı H1-H3 kuralları uygulandı.
Çıktılar: `results_synthetic_rev_v13`, `results_synthetic_rev_v12`.

**v13, `common_scored`, line_z:**

| Yöntem | AUROC | AP | FPR90 |
|---|---|---|---|
| shadow_cc | **0,8987** | **0,0539** | **0,2367** |
| shadow_nc | 0,8883 | 0,0514 | 0,2627 |
| delayed_trim10_cc | 0,8564 | 0,0478 | 0,3750 |
| delayed_trim05_cc | 0,8344 | 0,0421 | 0,4328 |
| official_erx | 0,8074 | 0,0396 | 0,4903 |
| official_rt_ck_rxd | 0,6328 | 0,0401 | 0,7989 |
| official_rx_bil | 0,6171 | 0,0400 | 0,8229 |
| official_lbl_ad | 0,5503 | 0,0154 | 0,9966 |
| official_rx_baseline | 0,5328 | 0,0417 | 0,9145 |
| frozen | 0,1098 | 0,0174 | 0,9792 |

| Hipotez | Gözlem | Karar |
|---|---|---|
| H1 | AUROC +0,091, AP +0,014, FPR90 −0,254 | **Destek** |
| H2 | Δ = 0,0539 − 0,0478 = +0,0061 | **Pişmanlıksız** |
| H3 | S = \|0,0412 − 0,0467\| = 0,0055 (v12 line_z) | **Bilgisiz** |
| H4 | AUROC, AP ve FPR90'da 1. | Betimsel |

**Permütasyon kontrolü (v12, betimsel):**

| Mod | shadow_cc | shadow_cc_permgate | AP'de geçilen |
|---|---|---|---|
| line_z | AUROC 0,899, AP 0,0526, FPR90 0,237 | AUROC 0,864, AP 0,0509, FPR90 0,322 | 11/15 |
| raw | AUROC 0,937, AP 0,249, FPR90 0,136 | AUROC 0,908, AP 0,101, FPR90 0,176 | 15/15 |

Ters yönde line_z modunda AP üzerinden zamanlama etkisi zayıf ve karışık (11/15, ΔAP +0,002). AUROC (+0,035) ve FPR90 (−0,085) Shadow lehine. Raw modda etki güçlü (15/15, ΔAP +0,148). İleri yöndeki line_z sonucuna (FPR90 0,343'e karşı 0,989) göre zamanlama etkisinin büyüklüğü tarama yönüne bağlı. Bu, zamanlama bulgusunun keşifsel kalması gerektiğini destekliyor.

**Diğer gözlemler:**

- Frozen line_z AUROC 0,110: ters yönde arka plan kayması, ileri yönden çok daha sert.
- Shadow skip oranı %78-80, ileri yönle aynı düzeyde.
- Resmi dedektörler makalenin "Flipped" sütununu yeniden üretiyor: RT-CK-RXD 0,631 (makale 0,632), RX-BIL 0,615 (0,615), LBL-AD 0,552 (0,550).
- RX Baseline'ın merkezli penceresi simetrik olduğu için bizde ileri ve ters yön birebir aynı (0,531). Makaledeki 0,527 değeri, resmi `Sampler`'ın ters modda ilk satırı 0. satırdan alması ve bunun maskeyle bir satırlık kayma yaratmasıyla tutarlı.
- ERX bizde 0,810, makalede 0,760 ± 0,022. Fark muhtemelen projeksiyon tohumundan kaynaklanıyor.

## 7. Genel değerlendirme (Synthetic, ileri ve ters yön)

- **H1** iki yönde de destekleniyor: Shadow, resmi ERX'i AUROC, AP ve FPR90'da tutarlı biçimde geçiyor.
- **H2** iki yönde de "pişmanlıksız": en iyi sabit kırpmaya göre AP farkı +0,006/+0,007, kazanç eşiğinin altında.
- **H3** iki yönde de bilgisiz: ön kayıttaki moderatör koşulu sağlanmadı.
- **H4:** Shadow, iki yönde de 10 yöntem arasında üç metrikte birinci.
- Zamanlama etkisi (permütasyon) raw modda iki yönde de güçlü (15/15). Line_z modunda ileri yönde güçlü, ters yönde zayıf. Keşifsel olarak raporlanmalı.

## 8. Kalan isteğe bağlı analizler

- Beach ve SNP ters yön (yalnızca dayanıklılık betimlemesi; geliştirme sahneleri).
