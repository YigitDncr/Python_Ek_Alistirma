import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

################################################################
# 1-Gerekli kütüphaneleri import ediniz. Ve ardından Telco Customer Churn veri setini okutunuz.
################################################################

df = pd.read_csv("2.Hafta/Datasets/Telco-Customer-Churn.csv")
df.head()

################################################################
# 2-Telco Customer Churn veri setinin Shape, Dtypes, Head,Tail, Eksik Değer, Describe bilgilerini elde ediniz.
################################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)

check_df(df)

################################################################
# 3-Veri setinin Gender sütununda gezen ve gender sütununun "Male" sınıfı ile karşılaşınca 0 aksi durumla
# karşılaşınca 1 basan bir comphrension yazınız ve bunu " num_gender " adında yeni bir değişkene atayınız
################################################################

df["num_gender"] = ["0" if "Male" in col else "1" for col in df["gender"]]
df[["gender", "num_gender"]].head()

################################################################
# 4-"PaperlessBilling" sütununun sınıfları içerisinde "Yes" sınıfı için "Evt", aksi durum için "Hyr"
# bastıran bir lambda fonksiyonu yazınız ve sonucu "NEW_PaperlessBilling" adlı yeni oluşturduğunuz sütuna
# yazdırınız. (lambda fonksiyonunu apply ile kullanabilirsiniz)
################################################################

df["NEW_PaperlessBilling"] = df["PaperlessBilling"].apply(lambda x: "Evt" if "Yes" in x else "Hyr").head()
df[["PaperlessBilling", "NEW_PaperlessBilling"]].head()

################################################################
# 5-Veri setinde "Online" ifadesi içeren sütunlar kapsamında sınıfı "Yes" olanlara "Evet", "No" olanlara
# "Hayır", aksi durumda "İnterneti_yok" şeklinde sınıfları tekrar biçimlendirecek kodu yazınız.
# Not: lambda içerisinde if elif else syntax error u ile karşılaşmamak adına başka bir def fonksiyonu ile
# "Yes" olanlara "Evet", "No" olanlara "Hayır", aksi durumda "İnterneti_yok" şeklinde sınıfları tekrar
# biçimlendirecek fonksiyonu dışarıda oluşturup bu fonksiyonu lambda içerisine uygulayabilirsiniz.
################################################################

online = [col for col in df.columns if "Online" in col]
df[online].head(12)

def kont(x):
    if "Yes" in x:
        x = "Evet"
    elif "No internet service" in x:
        x = "Interneti_yok"
    else:
        x = "Hayır"
    return x

for i, n in enumerate(online):
    x = df[n].apply(lambda x: kont(x)).head(12)
    df[n] = x

df[online].head(12)

################################################################
# 6-"TotalCharges" değişkeninin 30 dan küçük değerlerini bulunuz. Eğer hata alırsanız bu değişkeninin gözlemlerinin
# tipini inceleyiniz ve belirtilen sorgunun gelmesi için uygun olan tipe çevirerek sorguya devam ediniz.
################################################################

df["TotalCharges"].head()

# Mentor çözümü
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].isnull.sum()
df["TotalCharges"].dtype
df[df["TotalCharges"]<30].head()

# Kendi çözümüm
# ' ' bu şekilde boş olan değişkenlerden kaynaklı tip dönüşümü engelleniyordu onları bulup "0" olarak değişi gerçekleştirildi
for i, col in enumerate(df["TotalCharges"]):
    if col == ' ':
        df["TotalCharges"][i].update("0")
        print(col)

# bir önceki döngünde 0 olarak değiştirilen değişkenleri kontrol edildi
for col in df["TotalCharges"]:
    if col == "0":
        print(col)

# tip dönüşümü yapıldı
for col in df["TotalCharges"]:
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    print(type(col))

[col for col in df["TotalCharges"] if col < 30]

################################################################
# 7-Ödeme yöntemi "Electronic check" olan müşterilerin ortalama Monthly Charges değerleri ne kadardır?
################################################################

df.loc[(df["PaymentMethod"] == "Electronic check")].groupby(["PaymentMethod"]).agg({"MonthlyCharges": "mean"})
# Mentor çözümü
df.loc[df["PaymentMethod"] == "Electronic check", "MonthlyCharges"].mean()

################################################################
# 8-Cinsiyeti kadın olan ve internet servisi fiber optik ya da DSL olan müşterilerin
# toplam MonthlyCharges değerleri ne kadardır?
################################################################

df.loc[(df["gender"] == "Female")].groupby(df["InternetService"] == "No").agg({"MonthlyCharges": "sum"})
# Mentor çözümü
df.loc[(df["gender"] == "Female") & (df["InternetService"] == "Fiber optic") | (df["InternetService"] == "DSL"),"MonthlyCharges"].sum()

################################################################
# 9-Churn değişkeninde Yes olan sınıflara 1 , aksi durumda 0 basan lambda fonksiyonunu Churn değişkenine uygulayınız.
################################################################

df["Churn"] = df["Churn"].apply(lambda x: 1 if "Yes" in x else 0)
df["Churn"].head()

################################################################
# 10-Veriyi Contract ve PhoneService değişkenerine göre gruplayıp bu değişkenlerin sınıflarının
# Churn değişkeninin ortalaması ile olan ilişkisini inceleyiniz.
################################################################

df.groupby(["Contract", "PhoneService"]).agg({"Churn": "mean"})

################################################################
# 11- 10.soruda istenen çıktının aynısını pivot table ile gerçekleştiriniz.
################################################################

pd.pivot_table(df, values='Churn', index=['Contract'], columns=['PhoneService'], aggfunc=np.mean)

################################################################
# 12-tenure değişkeninin sınıflarını kategorileştirmek adına kendi belirlediğiniz aralıklara göre
# tenure değerlerini bölerek yeni bir değişken oluşturunuz. Aralıkları labels metodu ile isimlendiriniz.
################################################################
df["tenure"].head()
bins = [0, 12, 24, 36, df["tenure"].max()]
mylabels = ['New', 'Star', 'Loyal', 'Master']
df["NEWC_tenure"] = pd.cut(df["tenure"], bins, labels=mylabels)
df[["NEWC_tenure", "tenure"]].head(11)

# Mentor çözümü
df["NEWC_tenure"] = pd.cut(df["tenure"], [0, 10, 15, df["tenure"].max()], labels=['New', 'Star', 'Loyal'])
df[["NEWC_tenure", "tenure"]].head(11)