# Sonuç kaydı v2: Standart HAD sahneleri (ön kayıtlı)

**Tarih:** 19.10.2026
**Ön kayıt:** `PREREG_v2_TR.md`, SHA-256 `55E265EED1564C08A57DF9270967D7BE3E288B118F9C0E7C1B01C9A4C8BCAE0C`
**Manifest:** `manifest.csv`, SHA-256 `F6B7DC5FCE82D2F68E366D0F69BF4B355C3B561B90572B0F4CFB8AA4C86B351C`
**Sapmalar:** Yok. Yöntem, parametreler ve sahne kümesi ön kayıttaki gibi. Analizden önce klasördeki ön kayıt dosyası eski taslakla değiştirilmişti; doğru sürüm analizden önce yerine konup hash'lendi.
**Çıktılar:** `results_v18_suite` (ileri, birincil), `results_v18_suite_rev` (ters, ikincil)

## 1. Sahne bazında AUROC (Shadow cc / resmi ERX, 3 tohum ortalaması)

| Sahne | İleri: Shadow | İleri: ERX | Fark | Ters: Shadow | Ters: ERX | Fark |
|---|---|---|---|---|---|---|
| cat-island | 0,8803 | 0,8686 | +0,012 | 0,8564 | 0,8883 | −0,032 |
| gulfport | 0,9463 | 0,9268 | +0,020 | 0,9206 | 0,9125 | +0,008 |
| los-angeles-1 | 0,8252 | 0,8069 | +0,018 | 0,8025 | 0,8050 | −0,003 |
| los-angeles-2 | 0,9106 | 0,8997 | +0,011 | 0,8758 | 0,8732 | +0,003 |
| pavia | 0,8662 | 0,8715 | −0,005 | 0,8704 | 0,8581 | +0,012 |
| texas-goast | 0,9712 | 0,9614 | +0,010 | 0,9668 | 0,9553 | +0,012 |

## 2. Hipotez kararları

| Hipotez | İleri (birincil) | Karar | Ters (ikincil) | Karar |
|---|---|---|---|---|
| **H1 AUROC (birincil)** | 5/6 kazanç, medyan +0,011, p = 0,031 | **Destek** | 4/6, medyan +0,005, p = 0,34 | Destek yok |
| H1 AP (ikincil) | 2/6, medyan −0,011, Holm p = 0,66 | Destek yok | 3/6, medyan −0,007, Holm p = 0,58 | Destek yok |
| H1 FPR90 (ikincil) | 5/6, p = 0,031, Holm p = 0,063 | Holm sonrası anlamlı değil | 5/6, p = 0,031, Holm p = 0,063 | Holm sonrası anlamlı değil |
| **H2 AUROC (birincil)** | 3/6, medyan 0,000 | **Fark yok** | 3/6, medyan +0,000 | Fark yok |
| H2 AP | 1/6, medyan −0,004, p(pişmanlık) = 0,031 | **Pişmanlık** | 0/6, medyan −0,007, p(pişmanlık) = 0,016 | **Pişmanlık** |
| H2 FPR90 | 3/6 | Fark yok | 3/6 | Fark yok |
| H3 zamanlama | 1/6 sahne | Zayıf | 0/6 sahne | Yok |
| H4 ortalama AUROC sırası | Shadow 3,67 (Trim10 ile eşit, en iyi); ERX 6,0 | Betimsel | Shadow 4,0 (RT-CK-RXD 3,5 sonrası ikinci); ERX 5,67 | Betimsel |

## 3. Yorum

1. **Birincil hipotez, birincil analizde destekleniyor** (p = 0,031, 6 sahnenin 5'inde kazanç). **Ancak ters yönde tekrarlanmıyor** (p = 0,34). En büyük kayıp ters yönde cat-island'da (−0,032). Bu sahnenin değerlendirme penceresinde yalnızca 19 anomali pikseli var. Kanıt bu nedenle **zayıf ve yöne bağlı** olarak nitelendirilmelidir. ERX'e göre AUROC iyileşmesi küçüktür (medyan yaklaşık +0,01).
2. **FPR90**, ERX'e göre iki yönde de 6 sahnenin 5'inde iyileşiyor. Bu tutarlı bir yön, ama ön kayıttaki Holm düzeltmesinden sonra anlamlı değil.
3. **AP'de** Shadow ERX'i geçmiyor ve sahne bazında seçilen en iyi sabit kırpmaya göre iki yönde de küçük ama anlamlı biçimde geride (medyan −0,004 / −0,007). Bu seçim sabit kural lehine tutucu olduğu için pişmanlık biraz abartılı olabilir; yine de ön kayıt kuralına göre "pişmanlık" olarak raporlanır.
4. **Kapının kendine özgü katkısı** (en iyi sabit kurala göre kazanç, zamanlama etkisi) bu sahnelerde görülmüyor. Kısa ve küçük anomalili sahnelerde kazanç esas olarak kırpılmış güncellemeden geliyor.
5. **Post hoc açıklama (doğrulanmadı):** Bu sahnelerde satır başına kirlenme genellikle %10 kırpma payının altında. Kapının yararlı olduğu durum, yani kirlenmenin kırpma payını aşması (Beach'teki büyük panel), burada nadir. Ayrıca 100-150 satırlık sahnelerde 20 satırlık ısınma ve 80-130 değerlendirme satırı, ERX'in tasarlandığı uzun tarama koşullarını temsil etmiyor olabilir.

## 4. Makale için sonuç

- "Gecikmeli kırpmalı güncelleme ERX'i AUROC'ta iyileştirir" iddiası 9 sahnede tutarlı. Bağımsız kaynakta ön kayıtlı birincil testi geçiyor, ama ters yönde tekrarlanmıyor.
- "Kapı, sabit kırpmadan daha iyidir" iddiası yalnızca Beach'te destekleniyor. Diğer sahnelerde fark yok; standart sahnelerde AP'de küçük bir maliyet var.
- Bu sonuçlar ön kayıt taahhüdü gereği makalede eksiksiz raporlanır.
