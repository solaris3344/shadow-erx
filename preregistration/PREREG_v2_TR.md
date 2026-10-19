# Ön kayıt v2: Standart HAD sahnelerinde dondurulmuş Shadow kapısı

**Tarih:** 19.10.2026
**Durum:** Bu not, aşağıdaki sahnelerde hiçbir dedektör çalıştırılmadan önce yazılmıştır. Sonuçlar görüldükten sonra değiştirilmez. Bu dosyanın ve `manifest.csv` dosyasının SHA-256 değerleri `PREREG_v2_SHA256.txt` içinde kayıtlıdır.

## 1. Geçmiş

- Beach ve SNP geliştirme sahneleridir. Synthetic ilk ön kayıtla (PREREG_TR.md, SHA-256 668169E7…252BFCE2) test edildi.
- Synthetic sonrasında yapılan ρ/λ duyarlılık analizi, ρ ≤ 0,7'nin bazı sahnelerde daha iyi olduğunu gösterdi. **Buna rağmen yöntem değiştirilmeden kullanılır.**
- Bu nottaki sahnelerin hiçbirinde şimdiye kadar hiçbir yöntem çalıştırılmadı. (19.10.2026'da `run_v18_suite.py` bir kez boş manifest ile başlatıldı ve 0 sahne işledi; hiçbir sonuç üretilmedi.)

## 2. Sahneler

- **Kaynak:** GT-HAD deposundaki (github.com/NanWangAC/GT-HAD, `data` klasörü) 6 sahne. ABU deposuna erişilemediği için ek sahne kullanılmamıştır. Bu sahnelerin boyut ve bant sayıları ABU serisinin bir alt kümesiyle uyumludur; makalede bu şekilde belirtilir.
- **Sahne kümesi (sabit, `manifest.csv` ile birlikte hash'lenmiştir):**

| Sahne | Piksel/satır | Satır | Bant | Değerlendirme penceresindeki anomali pikseli |
|---|---|---|---|---|
| cat-island | 150 | 150 | 188 | 19 |
| gulfport | 100 | 100 | 191 | 60 |
| los-angeles-1 | 100 | 100 | 205 | 127 |
| los-angeles-2 | 100 | 100 | 205 | 87 |
| pavia | 150 | 150 | 102 | 31 |
| texas-goast | 100 | 100 | 204 | 67 |

- **Uygunluk kuralı (mekanik, `convert_had.py`):** ısınmadan sonra en az 50 değerlendirme satırı ve değerlendirme penceresinde en az bir anomali pikseli. Altı sahnenin hepsi uygundur. Anomali piksel sayıları yalnızca bu kural için okunmuştur; hiçbir yöntem çalıştırılmamıştır.
- **Tarama yönü:** dosyada saklandığı haliyle sütunlar (eksen 1). Ters yön ikincil analizdir.
- **Test gücü uyarısı:** 6 sahneyle tek yönlü Wilcoxon testinde p < 0,05 için işaretli sıra toplamının en az 19/21 olması gerekir. Yani gate ya 6 sahnenin hepsinde kazanmalı ya da yalnızca farkın mutlak değerce en küçük olduğu bir iki sahnede kaybetmelidir. H1'in reddedilmesi bu nedenle "etki yok" olarak değil, "6 sahneyle gösterilemedi" olarak yorumlanır.

## 3. Dondurulmuş yöntem ve protokol

| Öğe | Değer |
|---|---|
| Ana yöntem | `shadow_cc`, ρ = 0,8, λ = 0,05 (değişiklik yok) |
| Isınma / tampon uzunluğu | **Tüm yöntemler için 20 satır** (sahneler 100-150 satır olduğu için 99 kullanılamaz). Resmi dedektörlerde yalnızca `buffer_len` değişir; RX Baseline gecikmesi buna bağlı olarak 10 satır olur. |
| Diğer ERX ayarları | `config_detection.yaml`: d = 5, momentum 0,1, `normalise_md = True` |
| Referanslar | Resmi `erx`, `rx_baseline`, `lbl_ad`, `rt_ck_rxd`, `rx_bil`; `frozen`, `delayed_trim05_cc`, `delayed_trim10_cc`, `shadow_nc` |
| Değerlendirme penceresi | `common_scored`: tüm yöntemlerin puanladığı sütunlar |
| Projeksiyon tohumları | 0, 1, 2; sahne değeri = tohum ortalaması |
| Permütasyon kontrolü | Tohum başına 5 karıştırılmış kapı |
| Kod | `run_v18_suite.py` (bu nottaki kurallar kod içinde uygulanır) |

## 4. Hipotezler ve karar kuralları (sahneler arası)

Birim **sahnedir**. Her sahne için tek değer kullanılır (3 tohumun ortalaması).

**H1 (birincil). ERX'e göre iyileşme.**
- Test: AUROC farkı (gate − resmi ERX) üzerinde tek yönlü Wilcoxon işaretli sıra testi.
- Karar: p < 0,05 ise destek; aksi halde destek yok.
- İkincil metrikler AP ve FPR90 (FPR90'da düşük iyi) aynı testle raporlanır; Holm düzeltmesi uygulanır.
- Not: 6 sahneyle ulaşılabilecek en küçük p değeri 1/64 = 0,0156'dır (Bölüm 2'deki güç uyarısı).

**H2. En iyi sabit kırpmaya göre.**
- Her sahne ve metrik için Delayed Trim05 ile Delayed Trim10'dan iyi olanı seçilir. Bu, sabit kural lehine tutucu bir seçimdir.
- Karar (AUROC birincil):
  - tek yönlü test gate lehine p < 0,05 ise **kazanç**,
  - sabit kural lehine p < 0,05 ise **pişmanlık**,
  - hiçbiri değilse **fark yok**.

**H3 (betimsel). Zamanlama.** Shadow'un AP'de karıştırılmış kapıların en az %80'ini geçtiği sahne sayısı raporlanır.

**H4 (betimsel). Konum.** Sahneler arası ortalama AUROC sırası raporlanır.

## 5. İkincil analizler

- Ters tarama yönü, aynı kurallarla.
- AP, FPR90, düzeltilmiş AUC(D,τ) ve AUC(F,τ) sahne bazında betimsel olarak raporlanır.
- Yayımlanmış çevrimdışı yöntem sonuçları yalnızca ayrı bir "çevrimdışı üst sınır" sütununda verilir; akış yöntemleriyle aynı tabloda sıralanmaz.

## 6. Raporlama taahhüdü

- H1-H4'ün sonucu ne olursa olsun makalede raporlanır.
- Sahne bazındaki tüm sonuçlar ek materyalde verilir.
- Kod hatası çıkarsa yalnızca hata düzeltilir; düzeltme `DEVIATIONS_v2.md` dosyasına tarih ve gerekçeyle yazılır.
