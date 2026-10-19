# Ön kayıt: Shadow kapısının görülmemiş sahnede sınanması

**Tarih:** 18.10.2026
**Durum:** Bu not, Synthetic sahnesi indirilmeden ve hiçbir yöntem üzerinde çalıştırılmadan önce yazılmıştır. Sonuçlar görüldükten sonra bu dosya değiştirilmez. Hash değeri `PREREG_SHA256.txt` dosyasındadır.

## 1. Geliştirme geçmişi (dürüstlük beyanı)

- Beach ve SNP, v1-v13 boyunca yöntem tasarımında kullanıldı. Bunlar geliştirme sahneleridir; doğrulama sayılmaz.
- Synthetic sahnesi (Zenodo 13370799, AVIRIS, 600 piksel × 2400 satır × 90 bant) bu tarihe kadar hiçbir analizde kullanılmadı.
- Altyapı, ERX makalesindeki (Garske ve ark., TGRS 2025) Beach ve SNP AUROC değerlerini beş resmi dedektör için ±0,001 içinde yeniden üretti. ERX için fark, projeksiyon tohumu varyansı içinde kalıyor.

## 2. Dondurulmuş yöntem ve ayarlar

| Öğe | Değer |
|---|---|
| Ana yöntem | `shadow_cc` (tutarlılık düzeltmeli kırpma) |
| Aday güncellemeler | skip, Trim05, Trim10 (tek satır gecikmeli) |
| core_fraction / penalty | 0,80 / 0,05 |
| ERX ayarları | `config_detection.yaml`: buffer 99, d=5, momentum 0,1, `normalise_md=True` |
| Projeksiyon tohumları | 0, 1, 2 |
| Permütasyon tekrarı (v12) | 5 |
| Ana değerlendirme penceresi | `common_scored` (v13) |
| Ana normalizasyon | line_z (resmi tespit protokolü) |
| Referanslar | Resmi `erx`, `rx_baseline`, `lbl_ad`, `rt_ck_rxd`, `rx_bil` (config'e göre); `frozen`, `delayed_trim05_cc`, `delayed_trim10_cc` |

Hiçbir parametre Synthetic sonuçlarına göre değiştirilmeyecek. Çalıştırma bir kod hatası yüzünden yarıda kalırsa yalnızca hata düzeltilir, düzeltme `DEVIATIONS.md` dosyasına tarih ve gerekçeyle yazılır.

## 3. Önceden belirlenmiş hipotezler ve karar kuralları

Tüm değerler 3 tohum ortalamasıdır. Synthetic ileri yön (`--reverse` yok) birincil analizdir.

**H1. ERX'e göre iyileşme.** `shadow_cc`, `official_erx`'e göre AUROC'u ve AP'yi artırır, FPR90'ı düşürür.
- Destek: üç metrikte de aynı yönde iyileşme.
- Kısmi: bir veya iki metrikte iyileşme.
- Ret: hiçbirinde iyileşme yok.

**H2. Sabit kırpmaya göre pişmanlık.** Δ = AP(`shadow_cc`) − max(AP(`delayed_trim05_cc`), AP(`delayed_trim10_cc`)).
- Kazanç: Δ > +0,02
- Pişmanlıksız: −0,02 ≤ Δ ≤ +0,02
- Pişmanlık: Δ < −0,02

**H3. Zamanlama etkisi koşulludur.** Aday yayılımı S = |AP(`delayed_trim05_cc`) − AP(`delayed_trim10_cc`)|, v12 line_z çıktısından.
- S ≥ 0,02 ise tahmin: `shadow_cc`, 15 permütasyonun en az 12'sini AP'de geçer. Geçerse destek, geçmezse ret.
- S < 0,02 ise H3 bilgisizdir; zamanlama hakkında iddia kurulmaz.

**H4. Referans yöntemlere göre konum (betimsel).** `shadow_cc`'nin AUROC, AP ve FPR90 sıralaması ve ms/satır değeri raporlanır. Eşik veya üstünlük iddiası önceden kurulmamıştır.

## 4. İkincil analizler (doğrulayıcı sayılmaz)

- Synthetic ters yön (`--reverse`): aynı H1-H3 kuralları, ikincil kanıt olarak.
- Beach ve SNP ters yön: geliştirme sahneleri olduğu için yalnızca dayanıklılık betimlemesi.
- `official_full` penceresi ve raw normalizasyon: ek materyal.

## 5. Raporlama taahhüdü

- H1-H4'ün sonucu ne olursa olsun makalede raporlanır.
- Olumsuz sonuç, yöntemin sınırı olarak tartışılır; ek ayar denemesiyle "kurtarılmaz".
- HyperAD `calc_auc_dt` / `calc_auc_ft` hatası nedeniyle AUC(D,τ), AUC(F,τ), AUC_TD ve AUC_BS değerleri düzeltilmiş tanımla verilir; yayımlanmış değerlerle farkı ek bölümde gösterilir. Ana karşılaştırma AUROC, AP ve FPR90 ile yapılır.
