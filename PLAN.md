# E-Ticaret Satın Alma Tahmini ve Dinamik İndirim Optimizasyonu Projesi

## Revizyon Notu

- Bu sürüm, ilk planın `ecommerce_user_behavior_8000.csv` varsayımından çıkarılıp `customer_purchase_data.csv` veri setine tam uyarlanmış revize edilmiş halidir.
- İş hedefi korunmuştur: müşterinin satın alma olasılığını tahmin etmek ve bu skora göre indirim kararı için karar desteği üretmek.
- Ancak veri yapısı değiştiği için proje dili, özellik mühendisliği yaklaşımı, risk modeli ve savunma çerçevesi yeniden düzenlenmiştir.

## Belge Amacı

- Bu belge, veri madenciliği dersi kapsamında geliştirilecek projenin kod yazımına geçmeden önce oluşturulmuş ayrıntılı uygulama planıdır.
- Belgenin amacı, proje öneri raporu ile ara değerlendirme raporunu tek bir profesyonel yürütme planında birleştirmektir.
- Plan, sadece teknik geliştirme adımlarını değil; veri kalitesi doğrulamasını, iş hedefini, raporlama yaklaşımını, deney tasarımını ve sunum hazırlığını da kapsamaktadır.
- Belge özellikle uygulama aşamasında gereksiz tekrarları, plansız kodlamayı ve veri sızıntısı gibi kritik hataları önlemek için hazırlanmıştır.
- Buradaki yaklaşım, önce problemi doğru tanımlamak, sonra veri kaynağını doğrulamak, daha sonra da kontrollü biçimde modelleme ve karar destek katmanına geçmek üzerine kuruludur.
- Belge, mevcut CSV dosyasına ilişkin yapılan ilk denetim bulgularını da dikkate alır.
- Bu nedenle plan, raporlarda belirtilen varsayımları doğrudan kabul etmek yerine, önce sahadaki veri gerçekliğini doğrulayan bir çerçeve ile ilerlemektedir.
- Belge, ders projesinin ölçeğine uygun olmakla birlikte, kurumsal seviye yazılım ve analitik geliştirme disiplinleriyle tasarlanmıştır.
- Bu planda henüz kod yoktur.
- Bu plan, kod öncesi strateji ve yürütme dokümanıdır.

## Kullanılan Kaynaklar

- Kaynak 1: Proje öneri raporu.
- Kaynak 2: Proje ara değerlendirme raporu.
- Kaynak 3: Çalışma alanında kullanılacak `customer_purchase_data.csv` veri kümesi.
- Kaynak 4: `customer_purchase_data.csv` üzerinde yapılan teknik denetim, şema doğrulama ve iç tutarlılık analizi.
- Kaynak 5: Literatür kısmında referans verilen ensemble learning, SMOTE, SHAP ve dinamik fiyatlama/indirim yaklaşımı.

## Gereksinim Teyidi

- İstek net olarak sadece plan üretmektir.
- Bu aşamada kod yazılmayacaktır.
- Çıktı, markdown formatında bir dosya olacaktır.
- Dosyanın adı `PLAN.md` olacaktır.
- Plan kısa olmayacaktır.
- Plan her maddede detaylı olacaktır.
- Plan, iki raporu okuyup sentezleyerek hazırlanacaktır.
- Plan, veri seti olarak `customer_purchase_data.csv` dosyasını temel alacaktır.
- Plan, makine öğrenmesi ile satın alma tahmini problemine odaklanacaktır.
- Plan, dinamik indirim optimizasyonunu ayrı bir karar katmanı olarak ele alacaktır.
- Plan, veri ön işleme, özellik mühendisliği, model eğitimi, değerlendirme, açıklanabilirlik ve simülasyon başlıklarını içerecektir.
- Plan, veri sızıntısı, veri kalitesi ve sınıf dengesizliği gibi riskleri açık biçimde ele alacaktır.
- Plan, raporlarda geçen SMOTE, Random Forest, Gradient Boosting, XGBoost ve SHAP bileşenlerini değerlendirip uygun kullanım sırasını belirleyecektir.
- Plan, yalnızca akademik bir rapor üretmek için değil, sunumda savunulabilecek teknik bir yol haritası sunmak için hazırlanacaktır.
- Plan, haftalara veya fazlara ayrılmış şekilde uygulanabilir olacaktır.
- Plan, ölçülebilir teslimatlar ve kontrol noktaları içerecektir.
- Plan, veri bilimi tarafı ile iş etkisi tarafını birlikte dengeleyecektir.
- Plan, gerçek zamanlı iddiasını dikkatli şekilde çerçeveleyecektir.
- Çünkü mevcut veri kümesi clickstream değil; müşteri profili ve davranış özeti düzeyinde sinyal sunmaktadır.
- Bu yüzden plan, canlı üretim sistemi yerine önce çevrimdışı prototip ve karar simülasyonu hedefi koyacaktır.
- Plan, mevcut veri dosyasındaki tekrar kayıtlar ve sızıntı şüphesi taşıyan alanları ilk fazın merkezine alacaktır.
- Plan, proje kapsamı dışındaki işleri de açıkça yazacaktır.
- Plan, etik ve adalet risklerini ayrıca değerlendirecektir.
- Plan, değerlendirme metriklerini sadece doğruluk değil, iş değeri bağlamında da ele alacaktır.
- Plan, her fazda giriş koşulu, yapılacak iş, teslimat, risk ve çıkış kriteri mantığıyla yazılacaktır.
- Plan, raporlardaki güçlü yanları korurken zayıf varsayımları revize edecektir.
- Plan, 1000 satırdan uzun olacak şekilde ayrıntılandırılacaktır.

## İki Raporun Birlikte Okunmasından Çıkan Ana Sonuçlar

- Her iki raporun da ortak çekirdeği aynıdır: kullanıcının satın alma niyetini tahmin etmek.
- Her iki raporun da iş hedefi aynıdır: herkese indirim vermek yerine yalnızca kritik eşikteki kullanıcıya müdahale etmek.
- Her iki raporda da veri madenciliği problemi, ikili sınıflandırma problemi olarak çerçevelenmiştir.
- Her iki rapor, başlangıçta farklı ve daha davranış/oturum odaklı bir veri kümesini hedeflemektedir.
- Her iki rapor da veri ön işleme, eksik veri yönetimi, kategorik encoding, aykırı değer temizliği ve ölçekleme ihtiyacını kabul etmektedir.
- Her iki rapor da Random Forest ve Gradient Boosting ailesini güçlü adaylar olarak öne çıkarmaktadır.
- Her iki raporda da SHAP ile açıklanabilirlik önemli görülmektedir.
- Her iki rapor da nihai ürünün doğrudan canlı sisteme bağlanmasından önce bir simülasyon veya prototip aşaması öngörmektedir.
- Proje öneri raporu, iş problemini daha güçlü biçimde anlatmakta ve ticari mantığı daha iddialı kurmaktadır.
- Ara değerlendirme raporu ise deneysel yapı, test protokolü ve kıyaslama mantığını daha belirgin hale getirmektedir.
- Proje öneri raporunda dinamik indirim stratejisinin ticari anlamı geniş biçimde tartışılmıştır.
- Ara değerlendirme raporunda ise XGBoost daha baskın aday model olarak öne çıkmaktadır.
- Proje öneri raporunda clickstream ve oturum sonu sinyalleri güçlü biçimde kullanılmaktadır.
- Ancak yeni seçilen veri seti bu yapıyı birebir sağlamamaktadır; bunun yerine müşteri profili, geçmiş satın alma sayısı, sadakat programı ve web sitesinde geçirilen süre gibi özet sinyaller sunmaktadır.
- Bu nedenle planın ilk büyük kararı, rapor varsayımlarını olduğu gibi korumak değil, iş hedefini koruyup veri setine uygun yeni uygulama mimarisi kurmaktır.
- İki raporun da güçlü tarafı, sadece tahmin değil, eyleme dönük karar desteği hedeflemesidir.
- İki raporun da zayıf tarafı, indirim kararını doğrudan müdahale değişkeni ile karıştırma riskini yeterince tartışmamasıdır.
- Yeni veri setinde bu risk `DiscountsAvailed` sütunu üzerinden yeniden ortaya çıkmaktadır.
- Ayrıca raporlarda “gerçek zamanlı sistem” anlatısı güçlü olsa da veri seti yapısı gerçek zamanlı olay sıralaması sunmamaktadır.
- Bu yüzden nihai proje savunmasında “gerçek zamanlı üretim sistemi” yerine “gerçek zamanlıya yakın karar mantığını simüle eden çevrimdışı prototip” dili kullanılmalıdır.
- İki raporun birlikte okunmasından çıkan en doğru strateji şudur: önce veriyi doğrula, sonra sınıflandırma modelini kur, sonra açıklanabilirlik analizini yap, en son ayrı bir iş kuralı katmanıyla indirim politikasını simüle et.
- Bir başka kritik sonuç şudur: satın alma tahmini ile indirim etkisi tahmini aynı problem değildir.
- Satın alma olasılığı yüksek olan kullanıcıya indirim vermemek için yalnızca propensity model kullanmak mümkündür.
- Ancak “indirimin gerçekten dönüşümü artıracağı kullanıcıyı bulma” problemi nedensel etki veya uplift yaklaşımına daha yakındır.
- Veri seti bu ikinci problemi tam çözmek için sınırlı olabilir.
- Dolayısıyla ders projesinde bu ayrım dürüst biçimde açıklanmalıdır.
- Yeni veri seti, rapordaki oturum-içi pop-up anlatısından daha çok “satın alma eğilimine göre kampanya hedefleme” sistemine uygundur.
- Raporlarda geçen başarı beklentileri sunumda korunabilir, fakat bunlar “hedef performans” olarak sunulmalı, garanti çıktı gibi yazılmamalıdır.
- Rapor dili zaman zaman çok iddialıdır; plan dili daha ölçülü, test edilebilir ve savunulabilir olacaktır.

## Çalışma Alanındaki CSV Dosyasına Ait İlk Teknik Gözlem

- Çalışma alanındaki `customer_purchase_data.csv` dosyasında toplam 1500 satır bulunmaktadır.
- Veri seti toplam 9 sütundan oluşmaktadır: `Age`, `Gender`, `AnnualIncome`, `NumberOfPurchases`, `ProductCategory`, `TimeSpentOnWebsite`, `LoyaltyProgram`, `DiscountsAvailed`, `PurchaseStatus`.
- İlk denetimde eksik değer tespit edilmemiştir.
- Hedef değişken `PurchaseStatus` alanıdır ve iki sınıflıdır.
- İlk denetimde `PurchaseStatus='1'` sayısı 648, `PurchaseStatus='0'` sayısı 852 olarak gözlemlenmiştir.
- Bu dağılım yaklaşık olarak yüzde 43.2 pozitif, yüzde 56.8 negatif sınıfa karşılık gelmektedir.
- Bu nedenle veri seti aşırı dengesiz görünmemektedir; SMOTE gibi yöntemler varsayılan değil, yalnızca koşullu seçenek olarak ele alınmalıdır.
- Kategorik alanların tamamı sayısal kodlarla tutulmaktadır; `Gender` 0/1, `LoyaltyProgram` 0/1, `ProductCategory` 0-4 aralığında kodlanmıştır.
- Sayısal alan aralıkları ilk bakışta mantıklıdır: yaş 18-70, gelir yaklaşık 20 bin ile 150 bin arası, satın alma sayısı 0-20, sitede geçirilen süre yaklaşık 1-60 arasıdır.
- Veri setinde 1388 benzersiz satır görülmüş, 112 fazla tekrar satır tespit edilmiştir.
- Bu da yaklaşık yüzde 7.47 düzeyinde tekrar kayıt yükü bulunduğunu göstermektedir.
- Tekrar kayıtlar doğrudan veri sızıntısı yaratmaz; ancak train-test bölmesi öncesinde ele alınmazsa aynı örneklerin iki bölmeye dağılması sahte performans üretebilir.
- `DiscountsAvailed` ve `LoyaltyProgram` alanları hedef değişkenle güçlü ilişki göstermektedir.
- Bu durum veri setinin bilgi taşıdığını gösterirken aynı zamanda müdahale/sızıntı riski açısından dikkatli yorum gerektirdiğini de göstermektedir.
- Denetim sonucu, bu veri setinin eski bozuk veri kopyasına göre daha tutarlı bir temel sunduğunu, ancak “indirim etkisi” ile “satın alma eğilimi” problemlerinin ayrıştırılması gerektiğini göstermektedir.
- Denetim sonucu, proje planının veri bütünlüğü, tekrar kayıt temizliği ve müdahale değişkeni kontrolü etrafında yeniden kurulmasını gerektirmektedir.

## Veri Odaklı Kritik Uyarılar

- Bu veri setinde `user_id` bulunmamaktadır; bu yüzden kullanıcı seviyesinde grup bazlı bölme yapılamaz ve tekrar kayıtlar daha kritik hale gelir.
- `DiscountsAvailed` alanı ana risk değişkenidir; bu sütun geçmiş veya sonuçla ilişkili bir kampanya etkisini temsil ediyor olabilir.
- Kullanıcının kaç indirim kullandığını bilen bir model, sonradan indirim verilip verilmeyeceğini belirleyen politika için nedensel olarak güvenilir değildir.
- `LoyaltyProgram` alanı anlamlı olabilir; ancak bu alanın tahmin anında gerçekten bilinen ve operasyonel sistemde kullanılabilir bir bilgi olup olmadığı doğrulanmalıdır.
- `ProductCategory` alanı sayısal görünse de kategoriktir; 0-4 değerleri ordinal sayı gibi yorumlanmamalıdır.
- `Gender` alanı da sayısal kodlu kategorik değişkendir; nicel büyüklük gibi modele verilmemelidir.
- `AnnualIncome` ve `TimeSpentOnWebsite` alanları sinyal taşıyabilir; fakat veri setinin çok temiz ve kodlanmış yapısı, bu verinin simüle edilmiş veya ağır biçimde işlenmiş olabileceğini düşündürmektedir.
- Gerçek kullanım senaryosunda tahmin zamanı ile veri oluşum zamanı ayrımı yapılmalıdır.
- Eğer tahmin, kullanıcının siteyi gezerken üretilecekse `TimeSpentOnWebsite` alanının tam oturum sonu mu yoksa anlık ölçüm mü olduğu netleştirilmelidir.
- Eğer `DiscountsAvailed` satın alma kararı verildikten sonra oluşuyorsa bu değişken ana modelde kesinlikle kullanılmamalıdır.
- Oturum sonunda hesaplanan bir değişkeni, oturum ortasında kullanılacak model girdisi yapmak hedef sızıntısı yaratabilir.
- Bu nedenle planın veri denetimi fazında “tahmin anı veri erişilebilirlik matrisi” çıkarılacaktır.
- Proje savunmasında model türü kadar bu veri erişilebilirlik mantığını açıklamak da önemlidir.

## Proje Vizyonu

- Projenin temel vizyonu, müşteri profili ve davranış özeti verilerini kullanarak satın alma niyetini tahmin eden açıklanabilir bir karar destek prototipi üretmektir.
- Bu vizyonun teknik omurgası, denetlenmiş öğrenme ile müşteri/ziyaretçi düzeyinde satın alma olasılığı üretmektir.
- Bu vizyonun iş tarafındaki karşılığı, gereksiz indirimleri azaltırken riskli kullanıcıyı teşvik etmektir.
- Bu vizyonun ders projesi için doğru çerçevesi, çevrimdışı veri ile eğitilen ve örnek senaryolarda karar üreten bir simülasyon sistemidir.
- Bu vizyonun teslim edilebilir çıktıları; temiz veri süreci, kıyaslanmış modeller, açıklanabilirlik çıktıları, karar mantığı ve final sunumudur.

## Proje Başarı Tanımı

- Teknik başarı, yalnızca yüksek accuracy elde etmek değildir.
- Teknik başarı, veri sızıntısı içermeyen, tekrarlanabilir, açıklanabilir ve mantıklı bir model seçmektir.
- Teknik başarı, eğitim ve test ayrımının temiz yapılmasıdır.
- Teknik başarı, preprocessing işlemlerinin sadece eğitim verisine öğrenilip test verisine uygulanmasıdır.
- Teknik başarı, dengesiz sınıf varsa uygun metriklerle model değerlendirilmesidir.
- Teknik başarı, olasılık skorlarının kalibrasyonunun kontrol edilmesidir.
- Teknik başarı, örnek bazlı senaryolarda model kararlarının anlamlı açıklamalarla gösterilebilmesidir.
- İş başarısı, herkese indirim vermek yerine müdahale gerektiren segmentin belirlenmesidir.
- İş başarısı, gereksiz indirim oranını düşürmeye yönelik mantığın kurulmasıdır.
- İş başarısı, kar marjı koruma ile dönüşüm artışı arasında dengeli bir politika önerilebilmesidir.
- Akademik başarı, metodolojinin açık, tutarlı ve savunulabilir şekilde sunulmasıdır.
- Akademik başarı, rapordaki iddialar ile deney tasarımı arasında uyum kurulmasıdır.
- Akademik başarı, veri dosyasındaki tutarsızlıkların dürüstçe raporlanmasıdır.

## Proje Kapsamı

- Kapsam içi: Veri setinin temizlenmesi.
- Kapsam içi: Veri denetimi ve kalite kontrol raporu.
- Kapsam içi: Açıklayıcı veri analizi.
- Kapsam içi: Özellik mühendisliği stratejisi.
- Kapsam içi: Baseline ve gelişmiş sınıflandırma modellerinin karşılaştırılması.
- Kapsam içi: İhtiyaç halinde dengesizlik çözümü için kontrollü deneyler.
- Kapsam içi: SHAP ve açıklanabilirlik analizi.
- Kapsam içi: İş kuralı tabanlı indirim karar katmanı tasarımı.
- Kapsam içi: Çevrimdışı karar simülasyonu.
- Kapsam içi: Final raporu ve sunum materyalleri.
- Kapsam dışı: Gerçek e-ticaret API entegrasyonu.
- Kapsam dışı: Canlı stream processing altyapısı.
- Kapsam dışı: Üretim ortamında otomatik kampanya tetikleme.
- Kapsam dışı: Online learning veya reinforcement learning ile canlı optimizasyon.
- Kapsam dışı: Gerçek müşteri kimliği ile çalışan ticari dağıtım sistemi.

## Temel Varsayımlar

- Veri seti, müşteri profili ve davranış özeti düzeyinde satın alma eğilimini modellemek için yeterli sinyal içermektedir.
- Veri setindeki alan açıklamaları doğru kabul edilmeyecek, faz 1 içinde doğrulanacaktır.
- Mevcut CSV dosyasındaki etiket dağılımı büyük ölçüde kullanılabilir görünse de, `DiscountsAvailed` alanının anlamı proje için kritik bir doğrulama konusu olacaktır.
- Ders projesinde amaç, endüstriyel seviyede tam üretim sistemi değil, endüstriyel disiplinle hazırlanmış prototiptir.
- Dinamik indirim kısmı, nedensel etkiden çok karar destek simülasyonu olarak çerçevelenecektir.
- Sunumda, metodolojik dürüstlük metrik kadar önemli olacaktır.

## Kısıtlar

- Veri seti statiktir.
- Olay zaman sıralaması sınırlıdır.
- Gerçek kullanıcı yolculuğu, sayfa bazlı gezinme, sepet davranışı, cihaz tipi, reklama tıklama akışı ve bounce rate gibi oturum ayrıntıları veri setinde bulunmamaktadır.
- Veri setinde müşteri kimliği bulunmadığı için kullanıcı bazlı grup ayrımı yapılamaz.
- `DiscountsAvailed` alanı müdahale veya sonuçla ilişkili ise, indirim karar modeli için doğrudan kullanılamayabilir.
- Gelir, kategori ve sadakat gibi alanlar mevcut olsa da ürün fiyatı, marj, kampanya maliyeti ve gerçek ciro alanları bulunmamaktadır.
- Bu nedenle tam ekonomik optimizasyon yerine vekil metriklerle çalışan bir politika kurulacaktır.
- Ders süresi nedeniyle üretim güvenlik altyapısı değil, tasarım disiplini önceliklenecektir.
- Veri setinin tek başına nedensel indirim etkisini kanıtlaması mümkün olmayabilir.

## Yönetim İlkeleri

- Önce veri doğrulaması yapılacak, sonra modelleme başlayacaktır.
- Hiçbir preprocessing adımı eğitim-test ayrımından önce veri öğrenmeyecek şekilde tasarlanacaktır.
- Kodlanmış kategorik alanlar sayısal büyüklük gibi değil, kategorik değişken gibi ele alınacaktır.
- Tam tekrar satırlar bölme öncesinde temizlenecek veya kontrollü biçimde ele alınacaktır.
- `PurchaseStatus` ana hedef değişken olacaktır.
- Etiket dağılımı doğrulanmadan SMOTE uygulanmayacaktır.
- `DiscountsAvailed` için veri sızıntısı ve müdahale değişkeni denetimi yapılacaktır.
- Her model için aynı veri bölme stratejisi korunacaktır.
- Son model, sadece en yüksek metrik veren model değil, en tutarlı ve en savunulabilir model olacaktır.
- İş kararı ile model skoru birbirinden ayrılacaktır.
- Yorumlanabilirlik, final teslimatın zorunlu parçası olacaktır.

## Yüksek Seviye Yol Haritası

- Faz 0: Proje başlatma ve karar çerçevesini dondurma.
- Faz 1: Veri denetimi ve bütünlük doğrulaması.
- Faz 2: Açıklayıcı veri analizi ve hipotez üretimi.
- Faz 3: Ön işleme ve özellik mühendisliği tasarımı.
- Faz 4: Baseline modelleme.
- Faz 5: Sınıf dengesizliği ve eşik stratejisi deneyleri.
- Faz 6: Gelişmiş modelleme ve hiperparametre optimizasyonu.
- Faz 7: Açıklanabilirlik ve güvenilirlik analizi.
- Faz 8: Dinamik indirim karar katmanı tasarımı.
- Faz 9: Prototip ve simülasyon senaryoları.
- Faz 10: Doğrulama, stres testi ve teslim öncesi kalite kontrol.
- Faz 11: Final raporu, sunum ve demo paketi hazırlığı.

## Faz 0 - Proje Başlatma ve Karar Çerçevesini Dondurma

### Adım 0.1 - Problem tanımını son kez netleştir
- Amaç: Projeyi tek cümlede savunulabilir hale getirmek.
- Neden: Model, simülasyon ve indirim mantığı birbirine karışırsa proje odağını kaybeder.
- Yapılacak iş: “Müşteri profili ve davranış özeti verilerinden satın alma eğilimini tahmin eden ve bu skoru iş kuralı ile indirim kararına çeviren çevrimdışı karar destek prototipi” tanımı resmileştirilecek.
- Kontrol noktası: Proje tanımı hem öneri raporuyla hem ara raporla çelişmeyecek ama daha teknik ve daha dürüst bir dile sahip olacak.
- Çıktı: Bir paragraf uzunluğunda resmi problem tanımı.
- Risk ve önlem: Problem fazla genişlerse canlı sistem, fiyat optimizasyonu ve causal inference aynı anda ele alınamaz; bu yüzden kapsam net çizilecektir.

### Adım 0.2 - Başarı kriterlerini teknik ve iş kriteri olarak ayır
- Amaç: “Başarılı proje” ifadesini ölçülebilir hale getirmek.
- Neden: Sadece F1 skoru ile iş etkisi anlatılamaz; sadece iş hikayesi ile de teknik başarı savunulamaz.
- Yapılacak iş: Teknik metrikler, iş metrikleri ve sunum kalitesi metrikleri ayrı ayrı tanımlanacak.
- Kontrol noktası: Her metriğin neden seçildiği rapora yazılabilecek kadar açık olmalıdır.
- Çıktı: Başarı matrisi ve kabul kriterleri listesi.
- Risk ve önlem: Tek bir metrik üzerinden model seçilmesi engellenecektir.

### Adım 0.3 - Proje kapsamını dondur
- Amaç: Teslim tarihine kadar tamamlanabilecek bir planla ilerlemek.
- Neden: Ders projelerinde fazla geniş kapsam genellikle yüzeysel sonuç üretir.
- Yapılacak iş: Canlı entegrasyon, reinforcement learning, gerçek zamanlı stream ve kampanya otomasyonu kapsam dışı ilan edilecek.
- Kontrol noktası: Kapsam dışı maddeler final raporda da aynı ifadeyle yer almalıdır.
- Çıktı: Kapsam içi ve kapsam dışı listesi.
- Risk ve önlem: Sunum sırasında ek soru gelirse “gelecek çalışma” olarak ayrıştırılacaktır.

### Adım 0.4 - Veri ve iş problemi arasındaki ilişkiyi netleştir
- Amaç: Veri seti hangi soruya cevap veriyor, hangi soruya cevap veremiyor bunu açıkça belirlemek.
- Neden: Satın alma tahmini ile indirim etkisini tahmin etmek farklı bilimsel problemlerdir.
- Yapılacak iş: Propensity model ile intervention policy arasındaki ayrım yazılı hale getirilecek.
- Kontrol noktası: “Model yüksek riskli kullanıcıyı işaret eder, indirim kararı ayrı iş kuralı katmanında verilir” mantığı kabul görecek şekilde açıklanmalıdır.
- Çıktı: Mimari karar notu.
- Risk ve önlem: Nedensel iddialar abartılmayacak, eldeki veriyle yapılabilecek olan ifade edilecektir.

### Adım 0.5 - Çalışma disiplinini belirle
- Amaç: Deneylerin, notların ve teslimatların karışmasını önlemek.
- Neden: Veri bilimi projelerinde tekrar üretilebilirlik dağılırsa son hafta savunulabilir sonuç kalmaz.
- Yapılacak iş: Klasör, dosya adlandırma, deney isimlendirme ve çıktı saklama yaklaşımı planlanacak.
- Kontrol noktası: Aynı deneyi bir hafta sonra yeniden üretmek teorik olarak mümkün olmalıdır.
- Çıktı: Çalışma düzeni kılavuzu.
- Risk ve önlem: Rastgele denemeler yerine kayıtlı deney yaklaşımı izlenecektir.

### Adım 0.6 - Proje günlüğü ve karar kayıtlarını başlat
- Amaç: Neyi neden yaptığını sonradan açıklayabilmek.
- Neden: Sunum sırasında “neden bunu seçtin” soruları, teknik içeriğin kendisi kadar önemlidir.
- Yapılacak iş: Her önemli karar için kısa karar kaydı tutulacak; örneğin neden SMOTE kullanıldı veya neden kullanılmadı gibi.
- Kontrol noktası: En az veri denetimi, model seçimi, eşik seçimi ve açıklanabilirlik kararı için ayrı kayıtlar bulunmalıdır.
- Çıktı: Karar günlüğü şablonu.
- Risk ve önlem: Hafızaya dayalı savunma yerine dokümante edilmiş savunma yapılacaktır.

## Faz 1 - Veri Denetimi ve Bütünlük Doğrulaması

### Adım 1.1 - Şema ve veri tiplerini doğrula
- Amaç: CSV dosyasındaki sütunların rapordaki sütunlarla birebir uyumunu görmek.
- Neden: Sütun tipi veya kolon kayması modelleme öncesi fark edilmezse tüm deneyler geçersiz hale gelebilir.
- Yapılacak iş: Her sütunun veri tipi, beklenen aralıkları ve benzersiz kategori değerleri çıkarılacak.
- Kontrol noktası: `Gender`, `ProductCategory`, `LoyaltyProgram` ve `PurchaseStatus` alanlarının sayısal kod mu yoksa gerçek ordinal değer mi olduğu netleştirilecek.
- Çıktı: Şema doğrulama tablosu.
- Risk ve önlem: Özellikle boşluk, `0.0` ve `1.0` biçimleri yüzünden kategorik alanlar yanlış yorumlanırsa veri tipi standartlaştırılacaktır.

### Adım 1.2 - Hedef değişken bütünlüğünü incele
- Amaç: `PurchaseStatus` alanının gerçekten güvenilir hedef etiket olup olmadığını doğrulamak.
- Neden: Bu veri setinde eksik hedef yoktur; asıl risk, etiketin anlamının yanlış yorumlanması veya müdahale değişkenleriyle karışmasıdır.
- Yapılacak iş: Etiket dağılımı, 0 ve 1 kodlarının anlamı, tekrar satırların hedefe göre dağılımı ve etiketin diğer güçlü alanlarla ilişkisi incelenecek.
- Kontrol noktası: `PurchaseStatus=1` değerinin kesin olarak satın alma gerçekleşti anlamına geldiği belgelenmelidir.
- Çıktı: Etiket bütünlüğü raporu ve “devam et / veri sürümünü değiştir” karar kapısı.
- Risk ve önlem: Etiket doğrulanmadan eğitim yapılmayacak.

### Adım 1.3 - Eksik veri profilini ayrıntılandır
- Amaç: Eksik verinin hücre bazında, satır bazında ve sütun bazında yapısını anlamak.
- Neden: “Yüzde 2 eksik veri” gibi genel ifade, pratikte yeterli bir ön işleme kararı üretmez.
- Yapılacak iş: Her sütun için eksik oranı yeniden doğrulanacak; beklenen sonuç sıfır eksiktir, fakat simülasyon girdileri için güvenli fallback stratejisi de ayrıca belirlenecektir.
- Kontrol noktası: Bu veri setinde gerçekten eksik değer olmadığı doğrulanırsa imputation ana iş değil, koruyucu pipeline davranışı olarak ele alınacaktır.
- Çıktı: Eksik veri doğrulama notu ve gerektiğinde kullanılacak imputation taslağı.
- Risk ve önlem: Veri seti temiz görünse bile manuel senaryo girişlerinde eksik alan gelmesi ihtimaline karşı savunmalı tasarım korunacaktır.

### Adım 1.4 - Tekrar kayıt ve tekilleştirme denetimi yap
- Amaç: Veri setindeki birebir aynı satırların eğitim ve test güvenilirliğini bozup bozmadığını anlamak.
- Neden: Bu veri setinde kimlik alanı yoktur; bu yüzden tam tekrar satırlar ana sızıntı kaynaklarından biri haline gelir.
- Yapılacak iş: Aynı özellik ve hedef kombinasyonuna sahip tekrar satırlar sayılacak, oranları çıkarılacak ve temizleme/koruma kararı verilecektir.
- Kontrol noktası: Train-test bölmesi öncesinde tekrar satırların ele alınmasına ilişkin açık politika belirlenmelidir.
- Çıktı: Tekrarlı kayıt denetim notu.
- Risk ve önlem: Aynı satırların iki farklı bölmeye dağılması sahte performans üreteceği için tekrar kayıtlar bölme öncesi temizlenecek veya tekilleştirilecektir.

### Adım 1.5 - Dağılım ve mantık kontrolü yap
- Amaç: Sayısal alanlarda fiziksel olarak anlamsız veya aşırı uç değerleri belirlemek.
- Neden: Bazı aykırı kayıtlar gerçek kullanıcıyı değil veri hatasını veya sentetik bozulmayı temsil edebilir.
- Yapılacak iş: `Age`, `AnnualIncome`, `NumberOfPurchases`, `TimeSpentOnWebsite` ve `DiscountsAvailed` alanları için aralık ve tutarlılık kontrolleri yapılacak.
- Kontrol noktası: Negatif, imkansız veya iş kurallarıyla çelişen değerler etiketlenecek.
- Çıktı: Veri mantık hatası listesi.
- Risk ve önlem: Tüm aykırı değerleri silmek yerine, gerçek iş davranışı ile veri hatası ayrılacaktır.

### Adım 1.6 - Veri sızıntısı ve zamanlama denetimi yap
- Amaç: Hangi özniteliklerin tahmin anında gerçekten elde olabileceğini belirlemek.
- Neden: Oturum sonuna ait bilgi ile oturum ortasında tahmin yapan model kurmak akademik olarak hatalıdır.
- Yapılacak iş: Her sütun için “tahmin anında mevcut”, “oturum sonunda türetilir”, “müdahale sonrası oluşur” sınıflaması yapılacak.
- Kontrol noktası: `DiscountsAvailed`, `LoyaltyProgram` ve `TimeSpentOnWebsite` alanlarının kullanım politikası netleşecektir.
- Çıktı: Özellik erişilebilirlik matrisi.
- Risk ve önlem: Gerekirse iki model kurulacak; biri tam oturum sonu tahmini, diğeri erken müdahale için sınırlı özellikli model.

### Adım 1.7 - Faz 1 kapanış kararı al
- Amaç: Modelleme aşamasına geçmek için verinin yeterince güvenilir olup olmadığını resmi olarak karara bağlamak.
- Neden: Bu karar alınmadan yapılacak her deney, yanlış veri temeli üzerinde inşa edilmiş olabilir.
- Yapılacak iş: Etiket durumu, eksik veri, tekilleştirme ve sızıntı denetimi tek bir denetim memorandumunda toplanacak.
- Kontrol noktası: “Go”, “Conditional Go” veya “No Go” kararı verilecektir.
- Çıktı: Veri denetim raporu.
- Risk ve önlem: Eğer etiket problemi çözülmezse proje çerçevesi revize edilerek “veri bütünlüğü analizi ve sınırlı prototip” sunumu hazırlanacaktır.

## Faz 2 - Açıklayıcı Veri Analizi ve Hipotez Üretimi

### Adım 2.1 - Hedef değişkenin temel görünümünü çıkar
- Amaç: Satın alma etiketinin genel profilini görselleştirmek.
- Neden: Modelleme öncesinde sınıf dağılımını, boş etiketleri ve segment bazlı etiket oranlarını anlamak gerekir.
- Yapılacak iş: Hedef dağılımı, eksik etiket oranı ve kategori bazlı hedef oranları çıkarılacak.
- Kontrol noktası: Sınıf oranı savunulabilir biçimde rapora girecek kadar net ifade edilmelidir.
- Çıktı: Hedef değişken özet grafikleri.
- Risk ve önlem: Yanlış veri sürümü riski devam ediyorsa grafikler “ön denetim” etiketiyle sunulacaktır.

### Adım 2.2 - Sayısal değişkenlerin tek değişkenli analizini yap
- Amaç: Her sayısal alanın dağılımını, çarpıklığını ve yayılımını anlamak.
- Neden: İmputation, scaling ve outlier politikası bu sonuçlara göre tasarlanacaktır.
- Yapılacak iş: Histogram, kutu grafiği, temel istatistikler ve çarpıklık ölçümleri hazırlanacak.
- Kontrol noktası: Hangi alanlarda log dönüşüm, winsorization veya robust yaklaşım gerekebileceği not edilecektir.
- Çıktı: Tek değişkenli sayısal analiz bölümü.
- Risk ve önlem: Sadece görsel yorum yapılmayacak; istatistiksel özetler de saklanacaktır.

### Adım 2.3 - Kategorik değişkenlerin davranış analizini yap
- Amaç: Cinsiyet, ürün kategorisi ve sadakat programı gibi alanların dağılımını görmek.
- Neden: Kategorik alanların dengesizliği model kararlılığını etkileyebilir.
- Yapılacak iş: Frekans tabloları ve kategori bazlı satın alma oranları çıkarılacak.
- Kontrol noktası: Nadir kategori, tutarsız kategori yazımı veya boş kategori paternleri not edilecektir.
- Çıktı: Kategorik öznitelik analiz çıktıları.
- Risk ve önlem: Kategori sayısı düşük olsa bile az gözlemlenen segmentler için yorum abartılmayacaktır.

### Adım 2.4 - Hedef ile ikili ilişki analizini yap
- Amaç: Hangi özelliklerin satın alma ile ilk bakışta nasıl ilişkilendiğini görmek.
- Neden: Model seçimi kadar özellik mühendisliğine yön verecek hipotezler bu aşamada oluşur.
- Yapılacak iş: Segment bazlı hedef oranları, korelasyonlar ve görsel ayrıştırma analizleri yapılacak.
- Kontrol noktası: `NumberOfPurchases`, `TimeSpentOnWebsite`, `AnnualIncome`, `LoyaltyProgram` ve `DiscountsAvailed` alanlarının yönü ve gücü kontrol edilecektir.
- Çıktı: Hedef ilişkileri raporu.
- Risk ve önlem: Korelasyonun nedensellik olmadığı raporda özellikle vurgulanacaktır.

### Adım 2.5 - Segment bazlı kullanıcı profilleri oluştur
- Amaç: Farklı yaş, gelir, ürün kategorisi ve sadakat segmentlerinde davranış farklılıklarını görmek.
- Neden: Prototipte kullanılacak örnek senaryoların gerçekçi olması gerekir.
- Yapılacak iş: genç vs olgun kullanıcı, düşük gelir vs yüksek gelir, sadakat programında olan vs olmayan ve kategori bazlı segmentler ayrı incelenecek.
- Kontrol noktası: Sunumda anlatılabilecek en az dört güçlü segment hikayesi çıkarılmalıdır.
- Çıktı: Segment hikaye kartları.
- Risk ve önlem: Örneklem küçük segmentlerde genelleme yapılmayacaktır.

### Adım 2.6 - Anomali ve uç kullanıcı oturumları incele
- Amaç: Aykırı davranışın iş açısından anlamlı mı yoksa veri hatası mı olduğunu görmek.
- Neden: Raporlarda Z-score ile temizlik önerilse de e-ticarette aykırı değer bazen değerli müşteri davranışı olabilir.
- Yapılacak iş: çok yüksek gelir, çok yüksek site süresi, çok düşük satın alma geçmişi ile yüksek satın alma etiketi veya tekrar eden satır paternleri gibi durumlar incelenecek.
- Kontrol noktası: “Sil”, “işaretle”, “robust modele bırak” şeklinde üçlü politika önerisi üretilecektir.
- Çıktı: Aykırı değer karar notu.
- Risk ve önlem: Agresif aykırı değer silme politikası varsayılan seçim olmayacaktır.

### Adım 2.7 - Hipotez listesi oluştur ve dondur
- Amaç: EDA aşamasını açık uçlu keşiften kontrollü hipotez listesine çevirmek.
- Neden: Modelleme aşamasında neyi doğrulamaya çalıştığın bilinirse deneyler odaklı ilerler.
- Yapılacak iş: En az on test edilebilir davranış hipotezi yazılacak.
- Kontrol noktası: Her hipotez için hangi grafik, hangi model metriği veya hangi SHAP çıktısının kullanılacağı not düşülecektir.
- Çıktı: Hipotez kataloğu.
- Risk ve önlem: Veri keşfi sırasında görülen her ilginç örüntü nedensel gerçek gibi sunulmayacaktır.

## Faz 3 - Ön İşleme ve Özellik Mühendisliği Tasarımı

### Adım 3.1 - Veri bölme stratejisini kesinleştir
- Amaç: Eğitim, doğrulama ve test ayrımını veri sızıntısız kurmak.
- Neden: Yanlış bölme, tüm model karşılaştırmasını anlamsız hale getirir.
- Yapılacak iş: Stratified split, group split veya holdout yaklaşımı veri yapısına göre seçilecek.
- Kontrol noktası: Tam tekrar satırlar temizlenmeden hiçbir split yapılmayacak; çünkü kimlik alanı yokken en büyük sızıntı kaynağı budur.
- Çıktı: Bölme stratejisi kararı.
- Risk ve önlem: Test verisi en başta ayrılıp dokunulmadan tutulacaktır.

### Adım 3.2 - Özellik rollerini tanımla
- Amaç: Her sütunun modeldeki rolünü netleştirmek.
- Neden: Bazı alanlar giriş, bazı alanlar meta veri, bazı alanlar potansiyel sızıntı olabilir.
- Yapılacak iş: Kullanılacak, dışlanacak, opsiyonel ve yalnızca analiz için tutulacak alanlar listelenecek.
- Kontrol noktası: `PurchaseStatus` hedef olacak; `DiscountsAvailed` ana modelden varsayılan olarak dışlanacak ve yalnızca kontrollü ablation/analiz amacıyla ele alınacaktır.
- Çıktı: Özellik rol matrisi.
- Risk ve önlem: Faz 1 bulguları değişirse bu matris revize edilecektir.

### Adım 3.3 - Eksik veri doldurma stratejisini tasarla
- Amaç: Eksik değerleri veri sızıntısı oluşturmadan tamamlamak.
- Neden: Tüm veride öğrenilen medyan veya mod, test bilgisi sızdırır.
- Yapılacak iş: Bu veri setinde eksik değer görünmediği için aktif imputation yerine güvenli fallback politikası ve manuel giriş senaryoları için varsayılan doldurma kuralları tasarlanacak.
- Kontrol noktası: Ana pipeline eksiksiz veride gereksiz dönüşüm yapmamalı; ancak simülasyon arayüzü eksik girişlere karşı dayanıklı olmalıdır.
- Çıktı: Koruyucu imputation planı.
- Risk ve önlem: Temiz veri seti yanıltıcı güven oluşturabileceğinden, kullanıcı giriş ekranında zorunlu alan validasyonu korunacaktır.

### Adım 3.4 - Kategorik encoding politikasını belirle
- Amaç: Kategorik değişkenleri modelin doğru okuyacağı forma dönüştürmek.
- Neden: Algoritma türüne göre one-hot, ordinal veya native categorical destek tercihleri değişebilir.
- Yapılacak iş: `Gender`, `ProductCategory` ve `LoyaltyProgram` alanları için lojistik regresyonda one-hot, ağaç tabanlı modellerde ise kontrollü kategorik temsil stratejileri belirlenecek.
- Kontrol noktası: Eğitim ve testte aynı kolon uzayı korunacaktır.
- Çıktı: Encoding tasarım notu.
- Risk ve önlem: Eğitimde görülmeyen kategori değerleri için güvenli fallback stratejisi tanımlanacaktır.

### Adım 3.5 - Ölçekleme gereksinimini model bazında ayır
- Amaç: Hangi model için StandardScaler gibi ölçekleme gerektiğini netleştirmek.
- Neden: Ağaç tabanlı modeller çoğu zaman scaling istemez, lojistik regresyon ister.
- Yapılacak iş: Pipeline içinde model bazlı preprocessing dalları tasarlanacak.
- Kontrol noktası: Her model için aynı veri kaynağından türeyen ama uygun preprocessing kullanan deney hattı kurulacaktır.
- Çıktı: Model bazlı preprocessing haritası.
- Risk ve önlem: Tek bir preprocessing hattını tüm modellere zorla uygulama hatasından kaçınılacaktır.

### Adım 3.6 - Aykırı değer politikasını yeniden çerçevele
- Amaç: Uç değerlerle ilgili silme yerine kontrollü strateji kullanmak.
- Neden: E-ticaret verisindeki uç davranışlar bazen gerçek ve değerli kullanıcı sinyali olabilir.
- Yapılacak iş: Z-score sadece keşif aracı olarak kullanılacak; silme, clipping veya robust model tercihleri ayrı test edilecek.
- Kontrol noktası: Aykırı değer temizliğinin model performansına etkisi deneysel olarak ölçülecektir.
- Çıktı: Outlier karar şeması.
- Risk ve önlem: Tek bir heuristik ile toplu silme yapılmayacaktır.

### Adım 3.7 - Özellik mühendisliği adaylarını belirle
- Amaç: Ham değişkenlerden daha güçlü davranışsal sinyal türetmek.
- Neden: Etkileşimler çoğu zaman tekil alanlardan daha bilgilidir.
- Yapılacak iş: `income_bucket`, `purchase_frequency_bucket`, `time_spent_bucket`, `income_per_purchase_proxy`, `loyalty_time_interaction`, `category_income_interaction`, `high_time_low_history_flag`, `non_loyal_high_time_flag` gibi türetilmiş alanlar aday olarak listelenecek.
- Kontrol noktası: Türetilen her özellik için iş anlamı ve sızıntı riski ayrıca değerlendirilecektir.
- Çıktı: Özellik mühendisliği backlog’u.
- Risk ve önlem: Gösterişli ama anlamsız türevlerden kaçınılacaktır.

### Adım 3.8 - Ön işleme tasarımını küçük bir doğrulama deneyiyle kontrol et
- Amaç: Tasarlanan preprocessing hattının teknik olarak mantıklı sonuç verip vermediğini görmek.
- Neden: Büyük model tuning öncesi hatalı pipeline fark edilirse ciddi zaman kaybı önlenir.
- Yapılacak iş: Basit bir baseline model ile preprocessing seçenekleri hızlıca denenip veri kaçışı ve shape sorunları kontrol edilecek.
- Kontrol noktası: Train ve test kolon hizası, boş değer kalmaması ve mantıklı temel skor elde edilmesi doğrulanacaktır.
- Çıktı: Onaylı preprocessing hattı.
- Risk ve önlem: Bu adım küçük ölçekli olacak; hiperparametre optimizasyonuna dönüşmeyecektir.

## Faz 4 - Baseline Modelleme

### Adım 4.1 - Kör tahmin ve sınıf çoğunluğu baseline’ı kur
- Amaç: Her gelişmiş modelin gerçekten değer üretip üretmediğini görmek.
- Neden: Çok dengesiz veri setlerinde bazı gelişmiş modeller, basit majority baseline’dan anlamlı biçimde ayrışmayabilir.
- Yapılacak iş: Majority class, rastgele tahmin ve basit kural bazlı benchmarklar çıkarılacak.
- Kontrol noktası: Özellikle precision, recall, PR-AUC ve balanced accuracy karşılaştırılacaktır.
- Çıktı: Baseline referans tablosu.
- Risk ve önlem: Accuracy tek başına kullanılmayacaktır.

### Adım 4.2 - Lojistik regresyon baseline’ını kur
- Amaç: Basit, açıklanabilir ve olasılık üreten bir temel model elde etmek.
- Neden: Lojistik regresyon, daha karmaşık modellere geçmeden önce problemin lineer ayrışabilirlik seviyesini gösterir.
- Yapılacak iş: Düzenlileştirme seçenekleri ile temel lojistik model eğitilecek.
- Kontrol noktası: Katsayı yönleri, mantıklı iş etkileri ve olasılık kalitesi incelenecektir.
- Çıktı: Baseline lojistik model sonuçları.
- Risk ve önlem: Veri dengesizliği çok uç ise lojistik tek başına yanıltıcı olabilir; bu yüzden ek metrikler zorunlu tutulacaktır.

### Adım 4.3 - Basit ağaç tabanlı model kur
- Amaç: Doğrusal olmayan ilişkileri hızlıca görmek.
- Neden: E-ticaret davranışı çoğu zaman doğrusal değildir; ağaçlar ilk ayrışmayı gösterebilir.
- Yapılacak iş: Sınırlı derinlikte karar ağacı veya minimal Random Forest deneyi yapılacak.
- Kontrol noktası: Overfitting belirtileri ve feature importance davranışı incelenecektir.
- Çıktı: Ağaç bazlı ilk benchmark.
- Risk ve önlem: Tek bir ağaç kararsız olabileceği için yorum dikkatli yapılacaktır.

### Adım 4.4 - Olasılık kalibrasyonu ihtiyacını ölç
- Amaç: Model skorlarını “satın alma eğilim skoru” olarak kullanabilmek için olasılık kalitesini anlamak.
- Neden: İndirim kararı, sadece sınıf tahmini değil, güvenilir olasılık ister.
- Yapılacak iş: Calibration curve, Brier score ve reliability diagram çıkarılacak.
- Kontrol noktası: Yüksek skorların gerçekten yüksek olasılığa denk gelip gelmediği görülecektir.
- Çıktı: Kalibrasyon ön değerlendirmesi.
- Risk ve önlem: İyi sınıflandıran ama kötü kalibre olan model doğrudan politika motoruna bağlanmayacaktır.

### Adım 4.5 - Metrik setini nihai hale getir
- Amaç: Bundan sonraki tüm model karşılaştırmalarında sabit kullanılacak metrik setini dondurmak.
- Neden: Her model için farklı metrik konuşulursa objektif kıyas bozulur.
- Yapılacak iş: Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, Brier score ve confusion matrix resmi metrik seti olarak belirlenecek.
- Kontrol noktası: Sınıf dengesizliği durumunda hangi metriklerin karar verici olduğu ayrıca işaretlenecektir.
- Çıktı: Model kıyaslama şablonu.
- Risk ve önlem: Tek bir ana metrik yerine teknik ve iş hedefleri için ikili metrik seti kullanılacaktır.

### Adım 4.6 - Baseline fazı kapanış değerlendirmesi yap
- Amaç: Gelişmiş modellemeye geçmeden önce baseline sonuçlardan öğrenilenleri netleştirmek.
- Neden: Bu fazdan alınan dersler tuning stratejisini belirler.
- Yapılacak iş: Hangi değişkenler güçlü göründü, hangi metrikler sorunlu kaldı, hangi sızıntı şüpheleri devam ediyor bunlar özetlenecek.
- Kontrol noktası: Baseline raporu bir sayfada savunulabilir biçimde özetlenmiş olmalıdır.
- Çıktı: Baseline kapanış özeti.
- Risk ve önlem: Başarısız baseline da değerli bilgi olarak raporlanacaktır; saklanmayacaktır.

## Faz 5 - Sınıf Dengesizliği ve Eşik Stratejisi Deneyleri

### Adım 5.1 - Gerçek dengesizlik problemini yeniden doğrula
- Amaç: Dengesizlik stratejisini veri sürümü doğrulandıktan sonra karar vermek.
- Neden: Raporlardaki varsayım ile mevcut CSV aynı şeyi söylememektedir.
- Yapılacak iş: Nihai eğitim datasındaki etiket dağılımı yeniden hesaplanacak ve dengesizlik türü sınıflandırılacaktır.
- Kontrol noktası: Yeni veri setinde sınıflar yaklaşık dengeli göründüğü için SMOTE yalnızca sonuçlar gerçekten gerekliyse devreye alınacaktır.
- Çıktı: Dengesizlik karar notu.
- Risk ve önlem: Varsayıma dayalı SMOTE kullanımı yapılmayacaktır.

### Adım 5.2 - Class weight yaklaşımını test et
- Amaç: Sentetik örnek üretmeden önce daha düşük riskli çözümü değerlendirmek.
- Neden: Aşırı dengesiz veride sınıf ağırlıkları bazı modeller için daha güvenli olabilir.
- Yapılacak iş: Lojistik, Random Forest ve gradient boosting ailesinde class weight etkisi ölçülecek.
- Kontrol noktası: Recall artışı karşılığında precision nasıl değişiyor izlenecektir.
- Çıktı: Class weight deney özeti.
- Risk ve önlem: Sadece recall kazancı uğruna yanlış indirim riski büyüyorsa strateji reddedilecektir.

### Adım 5.3 - SMOTE uygulamasını sızıntısız kur
- Amaç: SMOTE kullanılacaksa bunun yalnızca eğitim katmanında ve çapraz doğrulama içinde çalışmasını sağlamak.
- Neden: Split öncesi SMOTE yapmak veri sızıntısı yaratır ve sahte başarı üretir.
- Yapılacak iş: Pipeline içine entegre edilmiş kontrollü SMOTE deneyi tasarlanacak.
- Kontrol noktası: Test verisi asla sentetik veri üretiminde kullanılmayacaktır.
- Çıktı: Sızıntısız SMOTE deney hattı.
- Risk ve önlem: Bu veri setinde dengesizlik orta düzeyde olduğu için SMOTE birincil değil, yalnızca koşullu ve karşılaştırmalı bir deney olarak tutulacaktır.

### Adım 5.4 - Kategorik yapı için uygun oversampling varyantını değerlendir
- Amaç: One-hot sonrası klasik SMOTE yerine daha uygun strateji gerekirse onu belirlemek.
- Neden: Kategorik alanlar için yanlış sentetik üretim anlamsız kombinasyonlar oluşturabilir.
- Yapılacak iş: SMOTENC veya alternatif yeniden örnekleme seçenekleri veri yapısına göre test edilecek.
- Kontrol noktası: Üretilen örneklerin iş anlamı korunuyor mu incelenecektir.
- Çıktı: Uygun oversampling seçimi.
- Risk ve önlem: Sentetik ama anlamsız müşteri profilleri model performansını yapay biçimde şişirirse yöntem terk edilecektir.

### Adım 5.5 - Eşik optimizasyonunu ayrı problem olarak ele al
- Amaç: Varsayılan 0.5 eşik yerine iş hedefine uygun karar eşiği bulmak.
- Neden: İndirim kararı binary sınıf değil, risk toleransı ile ilgilidir.
- Yapılacak iş: Precision-recall trade-off üzerinden farklı eşikler test edilecek.
- Kontrol noktası: “İndirimsiz bırak”, “izle”, “teklif göster” gibi çok bantlı karar yapısı düşünülecektir.
- Çıktı: Eşik ve karar bandı raporu.
- Risk ve önlem: Tek eşikli agresif politika yerine güven bandı yaklaşımı tercih edilebilir.

### Adım 5.6 - PR-AUC ve geri çağırım kalitesini öne çıkar
- Amaç: Özellikle nadir sınıf varsa gerçekten işe yarayan model davranışını görmek.
- Neden: ROC-AUC çok dengesiz veri setlerinde aşırı iyimser olabilir.
- Yapılacak iş: PR eğrileri, class-specific recall ve precision@k benzeri analizler yapılacak.
- Kontrol noktası: Hangi modelin kritik kullanıcıları daha iyi yakaladığı görülecektir.
- Çıktı: Dengesizlik duyarlı metrik paketi.
- Risk ve önlem: Sadece genel skor değil segment bazlı performans da incelenecektir.

### Adım 5.7 - Faz 5 sonunda iş açısından kabul edilebilir dengeyi seç
- Amaç: Teknik olarak yüksek ama iş açısından pahalı modelleri elemek.
- Neden: Yanlış kişiye indirim verme maliyeti, kaçan müşteriyi kurtarma faydası ile birlikte düşünülmelidir.
- Yapılacak iş: Teknik skorlar, yanlış pozitif maliyeti ve potansiyel kurtarma değeri birlikte yorumlanacak.
- Kontrol noktası: En iyi model değil, en iyi karar profili seçilecektir.
- Çıktı: Dengesizlik stratejisi kapanış kararı.
- Risk ve önlem: Fazla karmaşık ama küçük fayda sağlayan stratejiler sadeleştirilecektir.

## Faz 6 - Gelişmiş Modelleme ve Hiperparametre Optimizasyonu

### Adım 6.1 - Random Forest deney setini kur
- Amaç: Dayanıklı ve nispeten yorumlanabilir bir ensemble model elde etmek.
- Neden: Random Forest, kararlı benchmark sağlamak için uygundur.
- Yapılacak iş: Ağaç sayısı, derinlik, minimum yaprak boyu ve örnekleme parametreleri kontrollü şekilde taranacak.
- Kontrol noktası: Aşırı uyum ile performans arasındaki denge izlenecektir.
- Çıktı: Random Forest deney tablosu.
- Risk ve önlem: Çok geniş grid yerine mantıklı aralıklarla zaman kontrollü arama yapılacaktır.

### Adım 6.2 - Gradient Boosting ailesi için deney planı kur
- Amaç: Davranışsal örüntüleri daha güçlü yakalayabilen boosting yaklaşımını test etmek.
- Neden: Raporlarda en yüksek başarı beklentisi boosting tabanlı modellerdedir.
- Yapılacak iş: Sklearn Gradient Boosting veya benzeri temel boosting modeli kurulacaktır.
- Kontrol noktası: Learning rate ve estimator sayısı dengesi gözlemlenecektir.
- Çıktı: Temel boosting benchmark’ı.
- Risk ve önlem: Gereksiz derin modellerle aşırı öğrenme riski izlenecektir.

### Adım 6.3 - XGBoost veya CatBoost kullanım kararını ver
- Amaç: Ders projesine değer katacak ama gereksiz karmaşıklık yaratmayacak ileri model seçimini yapmak.
- Neden: Ara rapor XGBoost’u öne çıkarıyor; ancak veri kalitesi ve kurulum koşulları da hesaba katılmalıdır.
- Yapılacak iş: XGBoost uygulanabilirliği, CatBoost’un kategorik veri avantajı ve proje savunma değeri birlikte değerlendirilecek.
- Kontrol noktası: Kullanılan kütüphane seçim gerekçesi belgelenmiş olmalıdır.
- Çıktı: İleri model seçim notu.
- Risk ve önlem: Sadece popüler olduğu için değil, veri ve savunma uyumu nedeniyle seçim yapılacaktır.

### Adım 6.4 - Hiperparametre aramasını kontrollü yap
- Amaç: Tuning sürecini tekrarlanabilir ve kaynak kontrollü hale getirmek.
- Neden: Kör grid search zaman kaybı yaratır ve küçük veri setinde gereksiz olabilir.
- Yapılacak iş: Önce dar aralık, sonra gerekirse genişletilmiş aralık kullanan iki aşamalı tuning yapılacak.
- Kontrol noktası: Her tuning çalışması aynı split ve aynı metrik seti ile yürütülecektir.
- Çıktı: Tuning deney defteri.
- Risk ve önlem: Test verisi tuning için kullanılmayacaktır.

### Adım 6.5 - Olasılık kalibrasyonu uygula ve ölç
- Amaç: Seçilen ileri modelin güvenilir skor üretmesini sağlamak.
- Neden: İndirim politikası için kalibre edilmemiş skor risklidir.
- Yapılacak iş: Platt scaling veya isotonic calibration gibi yöntemler kontrollü biçimde denenecek.
- Kontrol noktası: Kalibrasyon sonrası teknik ve iş metriği dengesi tekrar ölçülecektir.
- Çıktı: Kalibre model varyantları.
- Risk ve önlem: Kalibrasyon sınıflandırma başarısını düşürürse bu etki açıkça raporlanacaktır.

### Adım 6.6 - Model dayanıklılığını veri dilimlerinde ölç
- Amaç: Tek bir ortalama skora aldanmadan modelin hangi segmentlerde zayıf olduğunu görmek.
- Neden: Yaş, gelir, sadakat programı ve ürün kategorisi segmentlerinde performans farklı olabilir.
- Yapılacak iş: Slice-based evaluation yapılacak.
- Kontrol noktası: Kritik segmentlerde çok düşük performans varsa model veya özellik seti revize edilecektir.
- Çıktı: Segment bazlı performans raporu.
- Risk ve önlem: Global başarı, yerel başarısızlıkları gizlemeyecektir.

### Adım 6.7 - Özellik stabilitesi ve varyansını incele
- Amaç: Modelin farklı fold’larda benzer mantıkla çalışıp çalışmadığını görmek.
- Neden: Çok oynak feature importance, modelin kararsız olduğuna işaret edebilir.
- Yapılacak iş: Fold bazlı önem sıraları ve performans varyansı incelenecek.
- Kontrol noktası: Kararsız model, tek seferlik yüksek skor üretse bile dikkatle değerlendirilecektir.
- Çıktı: Stabilite analizi.
- Risk ve önlem: Sadece en yüksek fold sonucu değil ortalama ve standart sapma raporlanacaktır.

### Adım 6.8 - Champion ve challenger model seçimini yap
- Amaç: Sonraki fazlarda açıklanabilirlik ve politika testleri için ana modeli belirlemek.
- Neden: Tüm modellerle sonsuza kadar ilerlemek proje odağını bozar.
- Yapılacak iş: Bir ana model ve bir yedek model seçilecek.
- Kontrol noktası: Seçim kararı; metrik, kalibrasyon, yorumlanabilirlik ve iş mantığı ile gerekçelendirilecektir.
- Çıktı: Champion/challenger kararı.
- Risk ve önlem: Fark çok küçükse daha basit ve daha savunulabilir model tercih edilecektir.

## Faz 7 - Açıklanabilirlik ve Güvenilirlik Analizi

### Adım 7.1 - Global feature importance analizi yap
- Amaç: Modelin genel olarak hangi sinyallere dayandığını görmek.
- Neden: İş birimi bakışında ilk sorulan konu, satın almayı neyin belirlediğidir.
- Yapılacak iş: Permutation importance, tree importance ve gerekirse coefficient analizi çıkarılacak.
- Kontrol noktası: `DiscountsAvailed` veya başka açık sızıntı alanları önem kazanıyorsa ana model reddedilecek veya özellik seti daraltılacaktır.
- Çıktı: Global önem sıralaması.
- Risk ve önlem: Tek bir importance yöntemi yeterli görülmeyecektir.

### Adım 7.2 - SHAP ile global açıklama üret
- Amaç: Özelliklerin tahmin yönüne ve büyüklüğüne etkisini daha şeffaf göstermek.
- Neden: Final sunumunda en güçlü anlatı araçlarından biri SHAP olacaktır.
- Yapılacak iş: Summary plot, dependence plot ve en etkili değişkenlerin yönlü analizi hazırlanacak.
- Kontrol noktası: Çıktılar, iş hikayesi ile çelişiyorsa veri sızıntısı ihtimali yeniden kontrol edilecektir.
- Çıktı: SHAP global analiz paketi.
- Risk ve önlem: SHAP çıktıları mutlak nedensellik gibi sunulmayacaktır.

### Adım 7.3 - Yerel açıklamalar ve örnek senaryolar üret
- Amaç: Tek tek kullanıcı senaryolarında modelin neden o kararı verdiğini göstermek.
- Neden: Simülasyon ekranında görülen kararın anlaşılır olması gerekir.
- Yapılacak iş: En az beş tipik ve üç atipik müşteri/ziyaretçi profili için local explanation üretilecektir.
- Kontrol noktası: Açıklamalar Türkçe iş diliyle anlatılabilir hale getirilecektir.
- Çıktı: Senaryo bazlı açıklama kartları.
- Risk ve önlem: Sadece modelin sevdiği örnekler değil, hata yaptığı örnekler de incelenecektir.

### Adım 7.4 - Adalet ve segment dengesi kontrolü yap
- Amaç: Modelin belirli demografik segmentlerde orantısız hata yapıp yapmadığını görmek.
- Neden: Dinamik indirim kararları yaş veya cinsiyet segmentlerinde adaletsiz görünmemelidir.
- Yapılacak iş: Yaş grubu, cinsiyet, sadakat programı ve ürün kategorisi bazlı hata oranları incelenecek.
- Kontrol noktası: Belirgin adaletsizlik varsa, bu durum raporda sınırlılık veya revizyon gereksinimi olarak işlenecektir.
- Çıktı: Fairness-lite değerlendirmesi.
- Risk ve önlem: Veri seti tam etik denetim için sınırlı olabilir; bu yüzden sonuçlar dikkatli yorumlanacaktır.

### Adım 7.5 - Açıklanabilirlik fazını iş kararıyla bağla
- Amaç: Teknik açıklamayı indirim politikasının mantığına bağlamak.
- Neden: Sunumda model ile politika arasındaki geçiş yumuşak kurulmalıdır.
- Yapılacak iş: “Hangi özellikler riski artırıyor ve neden müdahale düşündürüyor?” sorusuna cevap veren kısa yorum seti hazırlanacak.
- Kontrol noktası: Açıklanabilirlik çıktısı, politika motoruna girdi verecek kadar anlamlı olmalıdır.
- Çıktı: İşe çevrilmiş açıklama özeti.
- Risk ve önlem: Model açıklaması ile iş aksiyonu arasındaki bağ uydurma olmayacak, veriye dayalı olacaktır.

## Faz 8 - Dinamik İndirim Karar Katmanı Tasarımı

### Adım 8.1 - Satın alma tahmini ile indirim politikasını ayır
- Amaç: İki farklı problemi tek model içinde eritmemek.
- Neden: Satın alma olasılığı yüksek kullanıcıya indirim vermemek mantıklıdır; ancak indirim verildiğinde kimin döneceği ayrı problemdir.
- Yapılacak iş: Propensity skorunu üreten model ile aksiyon kuralını üreten politika mantığı ayrılacaktır.
- Kontrol noktası: Mimaride “skor üretimi” ve “aksiyon seçimi” iki ayrı katman olarak çizilecektir.
- Çıktı: Karar katmanı mimari diyagramı.
- Risk ve önlem: Nedensel etki iddiası gerekenden büyük kurulmayacaktır.

### Adım 8.2 - Müdahale aksiyon uzayını tanımla
- Amaç: Politika motorunun hangi çıktıları vereceğini netleştirmek.
- Neden: Tek seçenek “indirim ver / verme” yerine daha gerçekçi ve savunulabilir seçenekler gerekebilir.
- Yapılacak iş: “Müdahale etme”, “ürün önerisi göster”, “ücretsiz kargo öner”, “yüzde 10 indirim”, “yüzde 15 indirim” gibi aksiyon seti tanımlanacak.
- Kontrol noktası: Veri seti maliyet bilgisi taşımadığı için aksiyon seti sade ama anlamlı olacaktır.
- Çıktı: Aksiyon kataloğu.
- Risk ve önlem: Çok fazla aksiyon seçeneği sahte hassasiyet yaratabilir; sadeleştirilecektir.

### Adım 8.3 - Skor bantları oluştur
- Amaç: Olasılık skorunu doğrudan eyleme çevirecek basit ve anlatılabilir kurallar kurmak.
- Neden: Tek eşik çoğu zaman kaba kalır; risk bantları daha kontrollüdür.
- Yapılacak iş: Örneğin yüksek güven, orta risk, yüksek risk ve belirsiz alan şeklinde bantlar tasarlanacak.
- Kontrol noktası: Her bant için önerilen aksiyon ve gerekçe yazılacaktır.
- Çıktı: Skor bandı tablosu.
- Risk ve önlem: Bantlar keyfi değil, validation ve iş mantığına dayalı seçilecektir.

### Adım 8.4 - İş hedef fonksiyonunu tanımla
- Amaç: Politika kararını sadece sınıflandırma doğruluğuna değil, iş etkisine dayandırmak.
- Neden: Yanlış kişiye indirim vermek ile kaybedilen müşteriyi kurtarmak farklı ekonomik sonuç doğurur.
- Yapılacak iş: Vekil bir fayda fonksiyonu kurulacak; örneğin yanlış pozitif maliyeti ve kurtarılan dönüşüm faydası puanlanacaktır.
- Kontrol noktası: Gerçek ciro bilgisi olmadığı için kullanılan puanlama sistemi açıkça “proxy” olarak etiketlenecektir.
- Çıktı: Basit iş fayda modeli.
- Risk ve önlem: Gerçek finansal optimizasyon iddiası kurulmayacaktır.

### Adım 8.5 - `DiscountsAvailed` değişkeninin kullanım kuralını belirle
- Amaç: Müdahale değişkeni ile politika motoru arasında çelişki oluşmasını önlemek.
- Neden: Geçmişte indirim görmüş olmak, satın alma ile ilişkili olabilir ama yeni karar üretirken sızıntı veya confounding oluşturabilir.
- Yapılacak iş: Bu alan için üç ayrı senaryo test edilecektir: tamamen dışla, yalnızca analizde kullan, kontrollü ablation modeli içinde sınırla.
- Kontrol noktası: Hangi kullanım biçimi daha dürüst ve daha savunulabilir ise o seçilecektir.
- Çıktı: `DiscountsAvailed` kullanım kararı.
- Risk ve önlem: Politika motoru, geçmiş müdahale sonucunu yeni müdahale sebebi gibi kullanmayacaktır.

### Adım 8.6 - Karşı olgusal senaryo mantığını sade biçimde kur
- Amaç: “Bu kullanıcıya müdahale edilirse ne olabilir?” sorusuna en azından sezgisel cevap üretmek.
- Neden: Veri seti tam causal modeling desteklemeyebilir; yine de karar mantığına yardımcı senaryo üretilebilir.
- Yapılacak iş: Aynı skor bandındaki kullanıcılar için aksiyon öncesi ve sonrası varsayımsal senaryolar tasarlanacaktır.
- Kontrol noktası: Bu bölüm açıkça simülasyon olarak etiketlenecektir.
- Çıktı: Counterfactual-lite karar notu.
- Risk ve önlem: Gerçek deney verisi olmadan kesin uplift iddiası kurulmayacaktır.

### Adım 8.7 - Güvenlik ve kötüye kullanım korumalarını tasarla
- Amaç: Politika katmanının kullanıcı veya şirket aleyhine yanlış teşvikler üretmesini önlemek.
- Neden: Çok sık indirim gösteren politika, kullanıcıyı indirim bekleyen profile dönüştürebilir.
- Yapılacak iş: Maksimum indirim sıklığı, maksimum teklif düzeyi ve düşük güven skorlarında temkinli davranma kuralları yazılacak.
- Kontrol noktası: Politika, “karı koru” ve “kullanıcıyı kaçırma” arasında dengeli olmalıdır.
- Çıktı: Politika guardrail listesi.
- Risk ve önlem: Agresif indirim politikası özellikle yanlış pozitifleri pahalı hale getirir; sınırlar bu yüzden tanımlanacaktır.

### Adım 8.8 - Politika katmanını örnek senaryolarla doğrula
- Amaç: Kurulan karar mantığının sezgisel olarak doğru çalışıp çalışmadığını görmek.
- Neden: Teknik skor iyi olsa bile politika insan gözüne anlamsız gelebilir.
- Yapılacak iş: Yüksek niyetli sadık kullanıcı, yüksek web süresi ama düşük geçmiş satın alma gösteren kararsız kullanıcı, düşük gelirli ama sadakat programında olan kullanıcı ve düşük zaman/düşük geçmişe sahip ilgisiz kullanıcı senaryoları üzerinden politika test edilecektir.
- Kontrol noktası: Her senaryo için “neden bu aksiyon seçildi” açıklanabilir olmalıdır.
- Çıktı: Senaryo bazlı politika doğrulama tablosu.
- Risk ve önlem: İnsan sezgisi ile model çıktısı çatışıyorsa önce veri ve skor kalitesi tekrar kontrol edilecektir.

## Faz 9 - Prototip ve Simülasyon Senaryoları

### Adım 9.1 - Simülasyon amacını tanımla
- Amaç: Prototipin neyi göstereceğini netleştirmek.
- Neden: Prototip arayüzü, modelin kendisini değil karar değerini anlatmalıdır.
- Yapılacak iş: Girdi, skor, açıklama ve aksiyon önerisi akışı tanımlanacaktır.
- Kontrol noktası: Simülasyon, canlı entegrasyon vaadi vermeden işleyişi gösterecek seviyede tasarlanacaktır.
- Çıktı: Prototip kapsam notu.
- Risk ve önlem: Fazla geniş kullanıcı arayüzü geliştirme işinden kaçınılacaktır.

### Adım 9.2 - Senaryo kataloğu oluştur
- Amaç: Demo sırasında kullanılacak örnek kullanıcı profillerini önceden hazırlamak.
- Neden: Rastgele örnekler sunum akışını bozabilir.
- Yapılacak iş: En az sekiz tipik ve dört sınır durum senaryosu hazırlanacaktır.
- Kontrol noktası: Her senaryo bir iş hikayesine karşılık gelmelidir.
- Çıktı: Demo senaryo kataloğu.
- Risk ve önlem: Yalnızca modelin iyi çıktığı örnekleri seçmekten kaçınılacaktır.

### Adım 9.3 - Girdi doğrulama mantığını tasarla
- Amaç: Simülasyona elle girilecek değerlerin gerçekçi aralıkta kalmasını sağlamak.
- Neden: Gerçek dışı input verilirse model çıktısı da anlamsızlaşır.
- Yapılacak iş: Her alan için min, max, kategori seti ve zorunluluk kuralları tanımlanacaktır.
- Kontrol noktası: Eksik veya hatalı girişlere nasıl davranılacağı belirlenmelidir.
- Çıktı: Prototip input kuralları.
- Risk ve önlem: Kullanıcı girdisi validasyonu olmayan demo güven kaybettirir; bu yüzden kural seti şarttır.

### Adım 9.4 - Çıktı formatını standartlaştır
- Amaç: Simülasyon sonucunu teknik ama anlaşılır şekilde göstermek.
- Neden: Sadece yüzde skor göstermek çoğu zaman yeterli olmaz.
- Yapılacak iş: Satın alma skoru, risk bandı, önerilen aksiyon ve kısa gerekçe birlikte sunulacaktır.
- Kontrol noktası: Çıktı, bir öğretim üyesinin tek ekranda anlayabileceği kadar açık olmalıdır.
- Çıktı: Standart sonuç şablonu.
- Risk ve önlem: Fazla teknik detay tek ekrana yüklenmeyecektir; açıklama katmanlı verilecektir.

### Adım 9.5 - Kayıt ve izlenebilirlik yapısını tasarla
- Amaç: Demo sırasında hangi input için hangi kararın çıktığını kayıt altına almak.
- Neden: Sunum veya rapor sonrası aynı senaryoyu tekrar göstermek gerekebilir.
- Yapılacak iş: Senaryo adı, girdi özeti, skor, aksiyon ve açıklama alanlarından oluşan sonuç kaydı tasarlanacaktır.
- Kontrol noktası: Demo sonrası tablo halinde rapora aktarılabilir olmalıdır.
- Çıktı: Simülasyon log şeması.
- Risk ve önlem: Görsel demo yapılsa bile arka planda metinsel kayıt bulunacaktır.

### Adım 9.6 - Demo provasını tamamla
- Amaç: Sunum anında teknik akışın ve anlatının kesintisiz ilerlemesini sağlamak.
- Neden: Veri bilimi projelerinde en sık yaşanan sorun, modeli bilip gösterememektir.
- Yapılacak iş: Giriş, sonuç ve aksiyon anlatısı için prova yapılacaktır.
- Kontrol noktası: En fazla üç dakikalık kısa demo akışı ve beş dakikalık ayrıntılı demo akışı hazır olmalıdır.
- Çıktı: Demo senaryosu ve konuşma notları.
- Risk ve önlem: Prova yapılmayan arayüz veya senaryo final gösterimine çıkarılmayacaktır.

## Faz 10 - Doğrulama, Stres Testi ve Teslim Öncesi Kalite Kontrol

### Adım 10.1 - Uç durum test listesini oluştur
- Amaç: Model ve politika motorunun sınır koşullarda nasıl davrandığını görmek.
- Neden: Gerçek hayatta eksik, çelişkili veya sıra dışı girişler mutlaka olur.
- Yapılacak iş: çok düşük yaş, çok yüksek gelir, sıfır geçmiş satın alma, bilinmeyen kategori kodu, sadakat programı uyumsuzluğu ve aşırı uzun site süresi gibi vakalar test edilecektir.
- Kontrol noktası: Sistem sessizce yanlış davranmak yerine kontrollü sonuç üretmelidir.
- Çıktı: Edge case test matrisi.
- Risk ve önlem: Hatalı input karşısında sistemin nasıl tepki vereceği önceden belirlenecektir.

### Adım 10.2 - Hesaplama maliyeti ve performansı değerlendir
- Amaç: Ders projesi içinde yeterli hız ve açıklıkta çalışan çözüm seçmek.
- Neden: Aşırı ağır modeller küçük veri setinde gereksiz olabilir.
- Yapılacak iş: Eğitim süresi, tahmin süresi ve açıklama üretim süresi kabaca ölçülecek.
- Kontrol noktası: Simülasyon sırasında kabul edilebilir tepki süresi korunmalıdır.
- Çıktı: Performans değerlendirmesi.
- Risk ve önlem: SHAP hesaplaması pahalıysa demo için önceden hesaplanmış örnekler hazırlanacaktır.

### Adım 10.3 - Duyarlılık analizi yap
- Amaç: Küçük veri değişimlerinin model kararını aşırı oynatıp oynatmadığını görmek.
- Neden: Kararsız modeller, politika üretiminde güven kaybettirir.
- Yapılacak iş: Seçilmiş senaryolarda birkaç özellik küçük miktarda değiştirilip skor değişimi izlenecektir.
- Kontrol noktası: Mantıksız sıçramalar varsa model veya kalibrasyon gözden geçirilecektir.
- Çıktı: Sensitivity analizi.
- Risk ve önlem: Aşırı hassas modeller için karar bandı daha temkinli ayarlanacaktır.

### Adım 10.4 - Tekrarlanabilirlik kontrolü yap
- Amaç: Sonuçların aynı pipeline ile yeniden üretilebilir olduğunu göstermek.
- Neden: Akademik savunmada tekrar üretilebilirlik güven sağlar.
- Yapılacak iş: Rastgele tohumlar, veri bölme mantığı ve model konfigürasyonları sabitlenerek tekrar koşumu yapılacaktır.
- Kontrol noktası: Sonuçlar kabul edilebilir varyans içinde kalmalıdır.
- Çıktı: Reproducibility notu.
- Risk ve önlem: Çok oynak sonuç varsa model seçimi yeniden değerlendirilecektir.

### Adım 10.5 - Hata modları ve sınırlılıkları belgeye dök
- Amaç: Projenin neyi yapamadığını açıkça yazmak.
- Neden: Profesyonel proje anlatımı, güçlü yanlar kadar sınırlarını da dürüstçe gösterir.
- Yapılacak iş: müdahale değişkeni riski, tekrar satırlar, veri setinin işlenmiş/sentetik görünümlü yapısı, causal limitler ve ürün fiyatı eksikliği gibi sınırlar yazılacaktır.
- Kontrol noktası: Final raporda bu bölüm açık ve net bulunmalıdır.
- Çıktı: Limitasyon listesi.
- Risk ve önlem: Limitasyon saklanmayacak; aksine akademik olgunluk göstergesi olarak sunulacaktır.

### Adım 10.6 - Final kalite kapısını işlet
- Amaç: Teslim öncesi tüm teknik ve anlatısal parçaların hazır olduğundan emin olmak.
- Neden: Eksik bir grafik veya belirsiz bir karar gerekçesi final kaliteyi düşürür.
- Yapılacak iş: Veri, model, SHAP, simülasyon, rapor ve sunum için kapanış checklist’i uygulanacaktır.
- Kontrol noktası: Her başlık için “hazır / eksik / revizyon gerekli” durumu işaretlenecektir.
- Çıktı: Final readiness checklist.
- Risk ve önlem: Hazır olmayan kısım varsa kapsam küçültülerek daha sağlam teslimat yapılacaktır.

## Faz 11 - Final Raporu, Sunum ve Demo Paketi

### Adım 11.1 - Final rapor iskeletini netleştir
- Amaç: Teknik akışı raporda mantıksal sıraya oturtmak.
- Neden: Veri denetimi yapılmışsa bunun raporda görünmesi gerekir.
- Yapılacak iş: Giriş, literatür, veri denetimi, EDA, yöntem, deneyler, açıklanabilirlik, simülasyon, limitasyon ve sonuç bölümleri planlanacaktır.
- Kontrol noktası: Ara rapor ile final rapor arasında doğal evrim ilişkisi kurulacaktır.
- Çıktı: Final rapor içindekiler taslağı.
- Risk ve önlem: Sadece yöntem yazılıp veri problemleri atlanmayacaktır.

### Adım 11.2 - Görsel anlatı setini hazırla
- Amaç: Sunum ve rapor için tekrar kullanılabilir görseller üretmek.
- Neden: Aynı görselin farklı yerlerde tutarlı kullanılması profesyonel görünüm sağlar.
- Yapılacak iş: Hedef dağılımı, önemli öznitelikler, model kıyasları, SHAP grafikleri ve senaryo çıktıları seçilecektir.
- Kontrol noktası: Her görsel tek başına anlaşılabilir başlık ve kısa yorum taşımalıdır.
- Çıktı: Görsel kütüphanesi.
- Risk ve önlem: Fazla grafik yerine mesaj gücü yüksek grafikler seçilecektir.

### Adım 11.3 - Sunum hikayesini üç katmanda yaz
- Amaç: Farklı soru derinliklerine göre anlatımı yönetebilmek.
- Neden: Sunumda kısa özet, teknik detay ve iş etkisi farklı seviyelerde sorulabilir.
- Yapılacak iş: 30 saniyelik özet, 3 dakikalık teknik akış ve 10 dakikalık ayrıntılı savunma metni hazırlanacaktır.
- Kontrol noktası: Aynı hikaye tüm seviyelerde tutarlı kalmalıdır.
- Çıktı: Katmanlı anlatı notları.
- Risk ve önlem: Teknik olmayan dinleyici için sade, teknik dinleyici için derin cevaplar hazır olacaktır.

### Adım 11.4 - Demo ile rapor arasındaki bağı kur
- Amaç: Gösterilen senaryoların raporda da karşılığı olmasını sağlamak.
- Neden: Demo ile rapor birbirinden koparsa proje dağınık görünür.
- Yapılacak iş: Demo senaryoları raporun simülasyon bölümünde numaralandırılmış şekilde yer alacaktır.
- Kontrol noktası: Sunumda gösterilen her örnek raporda bulunmalı, rapordaki ana örnekler de demoya taşınmalıdır.
- Çıktı: Demo-rapor eşleme tablosu.
- Risk ve önlem: Sunum sırasında doğaçlama seçilen örnekler azaltılacaktır.

### Adım 11.5 - Gelecek çalışma bölümünü stratejik yaz
- Amaç: Projenin eksiklerini zayıflık gibi değil, mantıklı sonraki adım gibi sunmak.
- Neden: Üretim entegrasyonu ve causal uplift gibi konular sorulabilir.
- Yapılacak iş: Gerçek zamanlı veri akışı, A/B test, uplift modeling, maliyet bazlı optimizasyon ve canlı sistem entegrasyonu gelecekteki çalışmalar olarak yazılacaktır.
- Kontrol noktası: Bu bölüm mevcut teslimatı küçültmeden daha ileri vizyon göstermelidir.
- Çıktı: Gelecek çalışma listesi.
- Risk ve önlem: Gelecek çalışma bölümü, mevcut eksikleri gizlemek için kullanılmayacaktır.

### Adım 11.6 - Teslim paketini tamamla
- Amaç: Tüm çıktıları derli toplu şekilde sonlandırmak.
- Neden: Son gün aceleyle toplanan dosyalar kalite algısını düşürür.
- Yapılacak iş: Rapor, sunum, görseller, simülasyon açıklaması, karar günlüğü ve proje planı son kez gözden geçirilecektir.
- Kontrol noktası: Her dosya isimlendirme ve içerik açısından düzenli olmalıdır.
- Çıktı: Nihai teslim paketi.
- Risk ve önlem: Teslimden önce son dakika yapısal değişiklik yapılmayacaktır.

## Haftalara Bölünmüş Önerilen Takvim

### Hafta 1
- Problem tanımı dondurulur.
- Proje kapsamı netleştirilir.
- Başarı kriterleri yazılır.
- Kaynak raporlar ve veri kümesi eşlenir.
- CSV dosyası temel profili çıkarılır.
- Tekrar kayıt ve müdahale değişkeni riski için ilk not yazılır.
- Çalışma klasör planı hazırlanır.
- Karar günlüğü başlatılır.

### Hafta 2
- Şema doğrulama tamamlanır.
- Veri tipleri ve kategori değerleri kontrol edilir.
- Eksik veri profili ayrıntılandırılır.
- Tam tekrar satır denetimi yapılır.
- Hedef etiket bütünlüğü raporu hazırlanır.
- Veri sızıntısı potansiyeli ilk kez yazılı hale getirilir.
- “Go / No Go” ara kararı verilir.
- Gerekirse veri sürümü doğrulama çalışması yapılır.

### Hafta 3
- Tek değişkenli EDA yapılır.
- Kategorik dağılımlar incelenir.
- Hedef ile temel ilişkiler çıkarılır.
- Segment bazlı kullanıcı profilleri oluşturulur.
- Aykırı davranış hikayeleri ayrıştırılır.
- İlk hipotez listesi hazırlanır.
- Sunum için kullanılabilecek ilk görseller seçilir.
- EDA özeti yazılır.

### Hafta 4
- Veri bölme stratejisi kesinleşir.
- Özellik rol matrisi hazırlanır.
- İmputation stratejisi belirlenir.
- Encoding ve scaling politikaları netleşir.
- Aykırı değer politikası test planı yazılır.
- Özellik mühendisliği aday listesi oluşturulur.
- Ön işleme hattı küçük deneyle doğrulanır.
- Faz 3 kapanış notu yazılır.

### Hafta 5
- Majority baseline çalıştırılır.
- Lojistik regresyon baseline kurulur.
- İlk ağaç tabanlı baseline denenir.
- Metrik seti resmileştirilir.
- Kalibrasyon ihtiyacı ölçülür.
- Baseline sonuçları raporlanır.
- Zayıf ve güçlü sinyaller not edilir.
- İleri faz için tuning planı güncellenir.

### Hafta 6
- Etiket dağılımı son kez doğrulanır.
- Class weight yaklaşımı test edilir.
- SMOTE gerekiyorsa kontrollü şekilde denenir.
- Oversampling varyantları kıyaslanır.
- PR-AUC odaklı kıyaslama yapılır.
- Eşik optimizasyonu için ilk bantlar kurulmaya başlanır.
- İş maliyeti vekil modeli tasarlanır.
- Faz 5 kapanış kararı alınır.

### Hafta 7
- Random Forest tuning yapılır.
- Gradient Boosting tuning yapılır.
- XGBoost veya alternatif ileri model seçimi kesinleşir.
- Kontrollü hiperparametre araması yürütülür.
- Fold bazlı sonuçlar toplanır.
- Segment bazlı performanslar çıkarılır.
- Kararlı model adayları belirlenir.
- Champion ve challenger listesi oluşturulur.

### Hafta 8
- Kalibrasyon uygulanır.
- Kalibrasyon sonrası metrikler tekrar ölçülür.
- SHAP global analizleri çıkarılır.
- Yerel açıklama senaryoları hazırlanır.
- Adalet ve segment hataları incelenir.
- En güçlü üç iş içgörüsü seçilir.
- Faz 7 raporu yazılır.
- Sunum grafikleri güncellenir.

### Hafta 9
- Politika katmanı tasarlanır.
- Aksiyon uzayı belirlenir.
- Skor bantları netleşir.
- Proxy iş fayda fonksiyonu finalize edilir.
- `DiscountsAvailed` kullanım kararı netleştirilir.
- Guardrail listesi yazılır.
- Örnek politika senaryoları hazırlanır.
- Faz 8 kapanış notu yazılır.

### Hafta 10
- Simülasyon senaryo kataloğu hazırlanır.
- Girdi ve çıktı taslağı netleşir.
- Sonuç kayıt formatı oluşturulur.
- Demo senaryoları sıra ile çalışılır.
- Uç durum testleri uygulanır.
- Duyarlılık analizi yapılır.
- Tekrarlanabilirlik kontrol edilir.
- Teknik sınırlar raporlanır.

### Hafta 11
- Final rapor metni yazılır.
- Yöntem bölümü son haline getirilir.
- Deney sonuçları bölümü hazırlanır.
- SHAP ve karar katmanı yorumları eklenir.
- Simülasyon bölümü tamamlanır.
- Limitasyonlar dürüstçe yazılır.
- Gelecek çalışma bölümü hazırlanır.
- Sunum akışı yazılır.

### Hafta 12
- Demo provası yapılır.
- Sunum slaytları gözden geçirilir.
- Teslim paketi düzenlenir.
- Tüm grafikler son kez kontrol edilir.
- Metin içi tutarlılık denetlenir.
- Son kalite checklist’i uygulanır.
- Gereksiz kapsam kesilir, güçlü kısım korunur.
- Nihai teslim yapılır.

## Temel Teslimatlar Listesi

- Problem tanımı belgesi.
- Başarı kriterleri tablosu.
- Kapsam içi ve kapsam dışı listesi.
- Veri denetim raporu.
- Etiket bütünlüğü kararı.
- Eksik veri analizi çıktısı.
- Tekilleştirme kontrol notu.
- Veri sızıntısı değerlendirmesi.
- EDA özet raporu.
- Hipotez kataloğu.
- Özellik rol matrisi.
- İmputation strateji notu.
- Encoding tasarım notu.
- Ölçekleme politikası.
- Aykırı değer karar şeması.
- Özellik mühendisliği backlog’u.
- Baseline benchmark tablosu.
- Lojistik regresyon sonuç seti.
- Ağaç tabanlı benchmark.
- Kalibrasyon ön değerlendirmesi.
- Nihai metrik seti şablonu.
- Dengesizlik karar notu.
- Class weight deney özeti.
- SMOTE deney raporu.
- PR-AUC odaklı kıyas grafikleri.
- Eşik optimizasyon raporu.
- Proxy iş fayda modeli.
- Random Forest tuning tablosu.
- Boosting tuning tablosu.
- Champion model kararı.
- Challenger model kararı.
- Segment bazlı performans raporu.
- Stabilite analizi.
- SHAP global analiz görselleri.
- SHAP yerel senaryo kartları.
- Fairness-lite değerlendirmesi.
- İşe çevrilmiş açıklama özeti.
- Politika katmanı mimari notu.
- Aksiyon katalogu.
- Skor bandı tablosu.
- Guardrail listesi.
- Politika senaryo doğrulama tablosu.
- Simülasyon senaryo kataloğu.
- Girdi doğrulama kuralları.
- Sonuç ekranı şablonu.
- Simülasyon log şeması.
- Edge case test matrisi.
- Sensitivity analizi.
- Reproducibility notu.
- Limitasyon listesi.
- Final readiness checklist.
- Final rapor taslağı.
- Görsel kütüphanesi.
- Sunum konuşma notları.
- Demo-rapor eşleme tablosu.
- Gelecek çalışma listesi.
- Nihai teslim paketi.

## Kritik Karar Kapıları

### Karar Kapısı A - Veri sürümü doğru mu?
- Eğer şema, etiket anlamı ve tekrar kayıt politikası doğrulanırsa mevcut CSV ile devam edilecektir.
- Eğer veri setinin alan anlamları belirsiz kalırsa, lokal veri sözlüğü oluşturulmadan modelleme fazına geçilmeyecektir.
- Eğer `DiscountsAvailed` alanı müdahale sonrası oluşan değişken olarak netleşirse ana modelden kesin olarak çıkarılacaktır.

### Karar Kapısı B - `DiscountsAvailed` kullanılacak mı?
- Eğer bu alan müdahale sonrası oluşmuş bir değişkense tahmin modelinden çıkarılacaktır.
- Eğer alanın tahmin anında mevcut olduğu açık biçimde doğrulanırsa yalnızca kontrollü yan modelde test edilecektir.
- Eğer alan performansı çok artırıyor ama sızıntı şüphesi taşıyorsa ana modelde değil, raporda ablation sonucu olarak tutulacaktır.

### Karar Kapısı C - SMOTE gerekli mi?
- Eğer dengesizlik gerçek ve yönetilemez düzeydeyse SMOTE veya uygun alternatif kontrollü biçimde kullanılacaktır.
- Eğer class weight yeterliyse sentetik veri üretilmeyecektir.
- Eğer sentetik veri iş anlamını bozuyorsa oversampling terk edilecektir.

### Karar Kapısı D - Tek model mi, iki model mi?
- Eğer güvenli özellik seti ile yeterli performans elde edilirse tek ana model yeterli olacaktır.
- Eğer `DiscountsAvailed` alanının etkisini raporlamak gerekirse ana model + kontrollü ablation modeli yaklaşımı kullanılacaktır.
- Ders kapsamı nedeniyle iki tam işlevli model fazla gelirse, karar desteğine temel olacak güvenli ana model önceliklendirilecektir.

### Karar Kapısı E - Politika ne kadar iddialı olacak?
- Eğer veri sınırlıysa yalnızca “müdahale et / etme” gibi iki bantlı politika sunulacaktır.
- Eğer skor kalitesi ve proxy fayda modeli yeterli görünürse çok bantlı politika sunulacaktır.
- Her koşulda gerçek finansal optimizasyon iddiası yerine simülasyon tabanlı politika önerisi dili korunacaktır.

## Risk Kayıt Defteri

### Risk 1
- Risk: Yanlış veri sürümüyle çalışılması.
- Etki ve önlem: Tüm sonuçları geçersiz kılar; faz 1 içinde Kaggle açıklaması ve CSV saha bulguları eşleştirilerek doğrulama yapılacaktır.

### Risk 2
- Risk: `PurchaseStatus` alanının iş anlamının yanlış yorumlanması.
- Etki ve önlem: Tüm karar mantığını bozar; hedef değişken anlamı veri sözlüğü ve örnek dağılımlar üzerinden doğrulanacaktır.

### Risk 3
- Risk: `DiscountsAvailed` veya oturum sonu özet alanlarda veri sızıntısı olması.
- Etki ve önlem: Yapay yüksek skor üretir; tahmin anı erişilebilirlik matrisi hazırlanacaktır.

### Risk 4
- Risk: Sınıf dengesizliği varsayımının yanlış olması.
- Etki ve önlem: Gereksiz oversampling veya yanlış eşik seçimi doğar; dengesizlik türü ve oranı yeniden hesaplanacaktır.

### Risk 5
- Risk: Tekrar satırlar üzerinden ezberleme yapılması.
- Etki ve önlem: Sahte yüksek performans doğurur; tekrar satırlar bölme öncesi temizlenecektir.

### Risk 6
- Risk: Aykırı değerlerin agresif silinmesi.
- Etki ve önlem: Değerli kullanıcı sinyalleri kaybolabilir; silme yerine işaretleme ve robust yaklaşım öncelenecektir.

### Risk 7
- Risk: Accuracy’ye aşırı güvenilmesi.
- Etki ve önlem: Özellikle dengesiz veri setinde yanıltıcı olur; PR-AUC ve sınıf bazlı metrikler zorunlu tutulacaktır.

### Risk 8
- Risk: Kalibre olmayan skorlara göre indirim kararı verilmesi.
- Etki ve önlem: Yanlış kullanıcı segmentasyonu olur; kalibrasyon fazı zorunlu tutulacaktır.

### Risk 9
- Risk: Sunumda nedensel etki ile korelasyonun karıştırılması.
- Etki ve önlem: Akademik eleştiri doğurur; politika katmanı simülasyon olarak çerçevelenecektir.

### Risk 10
- Risk: Segment bazlı adaletsiz model davranışı.
- Etki ve önlem: Belirli gruplara haksız müdahale oluşabilir; slice-based evaluation yapılacaktır.

### Risk 11
- Risk: Aşırı geniş proje kapsamı.
- Etki ve önlem: Hiçbir bölüm derinleşmez; kapsam dışı maddeler sıkı korunacaktır.

### Risk 12
- Risk: Son hafta rapor, model ve demo arasında kopukluk olması.
- Etki ve önlem: Proje dağınık görünür; demo-rapor eşleme tablosu hazırlanacaktır.

### Risk 13
- Risk: Hiperparametre aramasının kontrolsüz büyümesi.
- Etki ve önlem: Zaman kaybı doğurur; iki aşamalı ve sınırlı tuning uygulanacaktır.

### Risk 14
- Risk: SHAP hesaplama maliyetinin demo akışını yavaşlatması.
- Etki ve önlem: Önceden hesaplanan örnekler hazırlanacak ve canlı hesaplama zorunlu tutulmayacaktır.

### Risk 15
- Risk: Veri setinin gerçek zamanlı iddia için yetersiz kalması.
- Etki ve önlem: Üretim sistemi söylemi yerine oturum düzeyi çevrimdışı prototip söylemi kullanılacaktır.

### Risk 16
- Risk: Eksik veri imputation stratejisinin hedef yanlılığı yaratması.
- Etki ve önlem: Eksiklik paternleri hedefle ilişkili ise eksik gösterge değişkeni de denenecektir.

### Risk 17
- Risk: Çok iyi görünen sonucun sızıntıdan kaynaklanması.
- Etki ve önlem: Aşırı yüksek skorlar özellikle sızıntı kontrolü ile yeniden incelenecektir.

### Risk 18
- Risk: Politika motorunun çok sık indirim önermesi.
- Etki ve önlem: Guardrail ve maksimum müdahale sınırları tanımlanacaktır.

### Risk 19
- Risk: Veri seti üzerinde ekonomik faydanın doğrudan ölçülememesi.
- Etki ve önlem: Proxy iş fayda modeli kullanılacak ve bunun sınırlı olduğu açıkça yazılacaktır.

### Risk 20
- Risk: Öğretim üyesinin “neden bu modeli seçtin” sorusuna yetersiz yanıt verilmesi.
- Etki ve önlem: Her ana karar için kısa karar kaydı tutulacaktır.

## Ölçüm ve Değerlendirme Matrisi

### Teknik Metrikler
- Accuracy yalnızca genel görünüm için kullanılacaktır.
- Precision, yanlış kullanıcıya indirim verme riskini okumak için önemlidir.
- Recall, satın alma potansiyeli olan kullanıcıları kaçırmama açısından izlenecektir.
- F1 skoru, precision ve recall dengesini özetlemek için kullanılacaktır.
- ROC-AUC genel ayrıştırma yeteneğini gösterecektir.
- PR-AUC dengesiz sınıf senaryosunda ana referans metriklerden biri olacaktır.
- Balanced accuracy, dengesizlik halinde adil kıyas için kullanılacaktır.
- Brier score olasılık kalitesini gösterecektir.
- Calibration curve skorların güvenilirliğini görselleştirecektir.
- Confusion matrix iş aksiyonlarının sayısal etkisini yorumlamak için tutulacaktır.

### İş Metrikleri
- Müdahalesiz bırakılan yüksek niyetli kullanıcı oranı izlenecektir.
- Müdahale önerilen riskli kullanıcı oranı izlenecektir.
- Yanlış müdahale oranı proxy maliyet olarak değerlendirilecektir.
- Kurtarılabilir kullanıcı segmentlerinin büyüklüğü izlenecektir.
- Skor bandı başına önerilen aksiyon dağılımı incelenecektir.
- Politika motorunun aşırı agresif veya aşırı pasif davranıp davranmadığı yorumlanacaktır.

### Akademik Metrikler
- Veri denetiminin açıkça raporlanması.
- Deney protokolünün tekrarlanabilir olması.
- Model seçiminin gerekçelendirilmiş olması.
- Limitasyonların dürüstçe belirtilmesi.
- Açıklanabilirlik çıktılarının rapora entegre edilmesi.
- Simülasyon kısmının yöntem ve sonuçlarla tutarlı olması.

## Özellik Mühendisliği Aday Havuzu

- `income_bucket` gelir segmenti.
- `purchase_frequency_bucket` geçmiş satın alma yoğunluğu segmenti.
- `time_spent_bucket` web sitesi etkileşim segmenti.
- `income_per_purchase_proxy` gelir ve geçmiş satın alma dengesi.
- `high_time_low_history_flag` kararsız ama ilgili kullanıcı göstergesi.
- `non_loyal_high_time_flag` sadakat düşükken ilgi yüksek kullanıcı göstergesi.
- `loyalty_time_interaction` sadakat ve site süresi etkileşimi.
- `number_of_purchases_bucket` geçmiş satın alma yoğunluğu segmenti.
- `age_bucket` demografik segment.
- `category_income_interaction` kategori ve gelir etkileşimi.
- `category_loyalty_interaction` kategori ve sadakat etkileşimi.
- `discount_usage_segment` yalnızca analiz amaçlı kampanya kullanım segmenti.
- `low_time_low_history_flag` düşük ilgi ve düşük deneyim göstergesi.

## Dışlanması Muhtemel Özellikler

- `DiscountsAvailed` tahmin anında güvenli olmadığı doğrulanırsa ana modelden dışlanacaktır.
- Tahmin anında bilinmeyen veya sonuç sonrası oluşan alanlar sınırlı kullanım alacaktır.
- Müdahale sonrası oluşan alanlar politika öncesi modelde kullanılmayacaktır.

## Faz Bazlı Çıkış Kriterleri

### Faz 0 çıkış kriterleri
- Problem tanımı yazılıdır.
- Başarı kriterleri yazılıdır.
- Kapsam netleşmiştir.
- Çalışma düzeni belirlenmiştir.

### Faz 1 çıkış kriterleri
- Veri denetim raporu tamamlanmıştır.
- Etiket bütünlüğü kararı verilmiştir.
- Eksik veri profili çıkarılmıştır.
- Sızıntı riski yazılı hale getirilmiştir.

### Faz 2 çıkış kriterleri
- EDA tamamlanmıştır.
- Hipotez listesi oluşturulmuştur.
- Önemli segment hikayeleri yazılmıştır.
- Aykırı değer politikasına giriş yapılmıştır.

### Faz 3 çıkış kriterleri
- Pipeline tasarımı onaylanmıştır.
- Özellik rol matrisi netleşmiştir.
- İmputation ve encoding planı hazırdır.
- Küçük doğrulama deneyi olumlu sonuç vermiştir.

### Faz 4 çıkış kriterleri
- Majority baseline çalıştırılmıştır.
- Lojistik regresyon baseline kurulmuştur.
- İlk ağaç modeli denenmiştir.
- Metrik şablonu dondurulmuştur.

### Faz 5 çıkış kriterleri
- Dengesizlik stratejisi kararı verilmiştir.
- Class weight ve gerekiyorsa SMOTE test edilmiştir.
- Eşik optimizasyon mantığı belirlenmiştir.
- İş maliyeti vekil çerçevesi kurulmuştur.

### Faz 6 çıkış kriterleri
- Champion model seçilmiştir.
- Challenger model seçilmiştir.
- Kalibrasyon değerlendirilmiştir.
- Segment bazlı performans incelenmiştir.

### Faz 7 çıkış kriterleri
- Global SHAP analizi vardır.
- Yerel açıklama senaryoları vardır.
- Adalet ve segment hata kontrolü yapılmıştır.
- Açıklamalar iş diline çevrilmiştir.

### Faz 8 çıkış kriterleri
- Politika katmanı tasarlanmıştır.
- Aksiyon uzayı belirlenmiştir.
- Skor bantları tanımlanmıştır.
- Guardrail listesi yazılmıştır.

### Faz 9 çıkış kriterleri
- Demo senaryoları hazırdır.
- Girdi/çıktı mantığı nettir.
- Simülasyon log yapısı belirlenmiştir.
- Demo provası yapılmıştır.

### Faz 10 çıkış kriterleri
- Edge case testleri tamamlanmıştır.
- Sensitivity analizi yapılmıştır.
- Reproducibility notu yazılmıştır.
- Limitasyon listesi bitmiştir.

### Faz 11 çıkış kriterleri
- Final rapor tamamlanmıştır.
- Sunum tamamlanmıştır.
- Demo akışı hazırdır.
- Teslim paketi düzenlenmiştir.

## Önerilen Klasör ve Dosya Organizasyonu

- `data/raw/` : Orijinal CSV dosyası ve kaynak tanımları.
- `data/processed/` : Temizlenmiş ve modelleme için hazırlanmış veri çıktıları.
- `notebooks/` : EDA ve kontrollü deney notebook’ları.
- `src/data/` : Veri okuma, şema kontrolü ve preprocessing bileşenleri.
- `src/features/` : Özellik mühendisliği mantığı.
- `src/models/` : Eğitim, tahmin, değerlendirme ve kalibrasyon akışları.
- `src/policy/` : İndirim karar katmanı ve iş kuralları.
- `src/simulation/` : Senaryo işleme ve prototip mantığı.
- `reports/figures/` : Grafikler ve SHAP çıktıları.
- `reports/tables/` : Sonuç tabloları.
- `docs/` : Karar günlükleri, metodoloji notları ve plan belgeleri.
- `tests/` : Veri şeması, preprocessing ve karar mantığı için birim test tasarımları.
- `artifacts/` : Eğitilmiş model dosyaları, kalibrasyon nesneleri ve deney kayıtları.
- `presentation/` : Slaytlar ve demo notları.

## Ders Projesi İçin Özel Savunma Stratejisi

- Sunuma doğrudan model adlarıyla değil, iş problemiyle başlanmalıdır.
- Hemen ardından herkese indirim vermenin neden kötü strateji olduğu açıklanmalıdır.
- Sonra veri setinin buna nasıl vekil sinyaller sunduğu anlatılmalıdır.
- Daha sonra çok kritik bir profesyonellik göstergesi olarak veri denetimi bulguları sunulmalıdır.
- Özellikle veri setinde tekrar kayıtlar bulunduğu ve `DiscountsAvailed` alanının sızıntı riski taşıdığı dürüstçe söylenmelidir.
- Bu, projeyi zayıflatmaz; aksine veri bilimi olgunluğunu gösterir.
- Ardından modelleme aşamaları, metrikler ve champion model anlatılmalıdır.
- Sonra SHAP ile modelin neden öyle düşündüğü gösterilmelidir.
- En sonunda skorun nasıl indirim kararına çevrildiği anlatılmalıdır.
- Demo, teoriyi destekleyen kısa ve kontrollü senaryolarla yapılmalıdır.
- “Gerçek zamanlı üretim sistemi yaptım” demek yerine “müşteri profili ve davranış özeti verisinde karar mantığını simüle ettim” denmelidir.
- “İndirim etkisini kesin ölçtüm” demek yerine “satın alma eğilimi skorunu iş kuralı ile aksiyona çevirdim” denmelidir.

## Öğretim Üyesinden Gerekirse Netleştirilebilecek Sorular

- Kullanılacak veri sürümünün doğruluğunu Kaggle ekran görüntüsü ile belgelemem istenir mi?
- Projede `DiscountsAvailed` alanını politika modelinde kullanmam uygun görülür mü, yoksa yalnızca analiz değişkeni olarak mı tutmalıyım?
- Ders kapsamında canlı arayüz bekleniyor mu, yoksa notebook tabanlı simülasyon yeterli mi?
- Sonuç değerlendirmesinde teknik metrik mi, iş hikayesi mi daha yüksek ağırlığa sahip?
- Veri setindeki etiket tutarsızlığı halinde raporda metodolojik revizyon yapılması kabul edilir mi?
- SHAP analizi görsel olarak zorunlu mu, yoksa tablo ve yorum da yeterli olur mu?
- İki model yaklaşımı mı, tek model artı iş kuralı yaklaşımı mı daha uygun görülür?
- Proxy iş fayda modeli kullanmam kabul edilir mi?
- Final teslimde kod kadar rapor kalitesine de aynı ağırlık verilecek mi?
- Sunum sırasında önceden hazırlanmış senaryolarla demo yapmak yeterli olacak mı?

## Proje İçin Önerilen Nihai Anlatı Cümlesi

- Bu proje, müşteri profili ve davranış özeti verilerini kullanarak kullanıcıların satın alma olasılığını tahmin eden, bu olasılığı açıklanabilir biçimde yorumlayan ve sonucu herkese indirim vermek yerine yalnızca riskli kullanıcıya hedefli kampanya önermeye dönüştüren bir karar destek prototipi geliştirmeyi hedeflemektedir.

## Son Özet

- Bu planın omurgası, raporların güçlü taraflarını koruyup varsayımlarını veri denetimi ile test etmektir.
- Planın en kritik farkı, veri bütünlüğünü modellemeden önce zorunlu bir faz haline getirmesidir.
- Plan, doğrudan model kurmak yerine önce “elimde doğru veri var mı” sorusunu cevaplar.
- Plan, satın alma tahmini ile indirim etkisi problemini birbirinden ayırır.
- Plan, metrik kadar açıklanabilirlik ve iş mantığına da önem verir.
- Plan, canlı sistem iddiasını kontrollü biçimde çevrimdışı simülasyon seviyesinde tutar.
- Plan, sunum sırasında savunulabilecek bir metodoloji üretmeyi hedefler.
- Plan, veri sızıntısı, sınıf dengesizliği, kalibrasyon ve fairness gibi gerçek veri bilimi risklerini görünür kılar.
- Plan, son haftaya bırakılmış dağınık bir proje değil, faz bazlı ilerleyen profesyonel bir ders projesi kurar.
- Kod aşamasına geçildiğinde bu belge, doğrudan yürütme rehberi olarak kullanılmalıdır.

## Hızlı Kontrol Listesi

- Problem tanımı donduruldu mu?
- Veri sürümü doğrulandı mı?
- Etiket bütünlüğü raporu yazıldı mı?
- Eksik veri profili çıkarıldı mı?
- Sızıntı riski değerlendirildi mi?
- EDA tamamlandı mı?
- Hipotez listesi oluşturuldu mu?
- Pipeline tasarımı onaylandı mı?
- Baseline modeller çalıştı mı?
- Dengesizlik stratejisi kararı verildi mi?
- Champion model seçildi mi?
- Kalibrasyon ölçüldü mü?
- SHAP analizi hazır mı?
- Politika katmanı yazılı olarak tasarlandı mı?
- Demo senaryoları hazır mı?
- Edge case listesi tamam mı?
- Limitasyonlar yazıldı mı?
- Sunum hikayesi hazır mı?
- Teslim paketi düzenlendi mi?

## Ek Ayrıntılı Uygulama Notları

- Eğer veri seti kaynağına ilişkin ek dokümantasyon bulunursa tüm EDA ve modelleme adımları bu sözlükle yeniden hizalanmalıdır.
- Eğer `DiscountsAvailed` alanı güvenli bulunmazsa ana model bu alan olmadan eğitilmelidir.
- Bu veri setinde eksik hedef etiketi bulunmadığı için ana kalite konusu eksiklik değil, tekrar kayıtlar ve alan anlamlarıdır.
- Tekrar satırlar train-test ayrımından önce ele alınmalıdır.
- Sınıf oranı aşırı dengesiz görünmediği için oversampling varsayılan çözüm olarak düşünülmemelidir.
- Aşırı uç dengesizlik gözlenirse anomaly detection veya class-weight yaklaşımı kısa bir alternatif olarak tartışılabilir.
- Ancak ders projesi ekseninin dağılmaması için ana akış supervised classification olarak kalmalıdır.
- Model skoru ile aksiyon arasına her zaman iş kuralı katmanı konmalıdır.
- Yüksek skor her zaman “hiçbir şey yapma” demek zorunda değildir; belirsizlik bandı varsa farklı izleme aksiyonu düşünülebilir.
- Orta skor bandı, indirim için en kritik karar alanıdır.
- Çok düşük skor bandında indirim vermek bazen gereksiz olabilir; çünkü kullanıcı çok ilgisiz olabilir.
- Bu nedenle politika motoru üç bantlı yaklaşım ile daha mantıklı kurulabilir.
- Sadık kullanıcı için düşük ama agresif olmayan teşvik, yeni kullanıcı için ise güven artırıcı mesaj önerilebilir.
- Proxy fayda fonksiyonunda “yanlış pozitif” maliyeti ile “doğru müdahale” faydası ağırlıklandırılmalıdır.
- Proxy fayda modeli gerçek finansal model değildir, ama karar mantığını savunmak için yeterli bir araç olabilir.
- Sunumda bir grafik mutlaka “gereksiz indirimden kaçınma” fikrini desteklemelidir.
- Bir başka grafik mutlaka “riskli kullanıcıyı yakalama” fikrini desteklemelidir.
- SHAP çıktılarında iş açısından en kolay anlatılan iki veya üç değişken öne çıkarılmalıdır.
- Çok fazla teknik detay, ana hikayeyi gölgelememelidir.
- Final raporda veri denetimi fazının bulunması projeyi daha olgun gösterecektir.
- Eğer öğretim üyesi yüksek başarı oranı sorarsa, önce veri sızıntısı kontrolü ve tekrar kayıt temizliği anlatılmalıdır.
- Eğer neden XGBoost seçildiği sorulursa yalnızca skor değil, yapısal gerekçe verilmelidir.
- Eğer neden lojistik regresyon hala kullanıldığı sorulursa açıklanabilir baseline olduğu söylenmelidir.
- Eğer neden SHAP yapıldığı sorulursa iş kararlarının şeffaflaştırılması için olduğu anlatılmalıdır.
- Eğer neden canlı sistem yapılmadığı sorulursa veri setinin müşteri özeti seviyesinde olduğu ve ders kapsamının prototip odaklı olduğu belirtilmelidir.
- Eğer neden indirim etkisini doğrudan ölçmediğin sorulursa veri setinin randomize deney veya uplift etiketleri taşımadığı açıklanmalıdır.
- Eğer veri seti çok temiz veya sentetik görünümlü bulunduysa, bu durum genellenebilirlik sınırlılığı olarak açıkça raporlanmalıdır.
- En önemli nokta, planın veri gerçekliğini kabul ederek ilerlemesidir.
- Bu yaklaşım, yüzeysel değil profesyonel bir proje izlenimi verecektir.

## Nihai Uygulama Sırası Tek Satırlık Özetler

- Önce veri doğru mu kontrol et.
- Sonra veri yapısını ve eksikleri çöz.
- Sonra EDA ile hipotez üret.
- Sonra güvenli preprocessing pipeline kur.
- Sonra baseline modelleri çalıştır.
- Sonra dengesizlik stratejisini dikkatle test et.
- Sonra güçlü ensemble modelleri tune et.
- Sonra skoru kalibre et.
- Sonra SHAP ile modeli açıkla.
- Sonra indirim karar katmanını ayrı tasarla.
- Sonra simülasyon senaryoları ile göster.
- Sonra kalite kontrol ve sunum hazırlığını tamamla.