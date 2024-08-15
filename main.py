import pandas as pd

# تحميل البيانات من ملف CSV
df = pd.read_csv('dataset/tmdb_5000_credits.csv')

# اختيار الأعمدة المهمة فقط
columns_to_keep = ['title', 'cast', 'crew']  # يمكنك تعديل الأعمدة حسب الحاجة
df_reduced = df[columns_to_keep]

# تقليل عدد الصفوف إلى عدد محدد (مثل 100 صف)
df_reduced = df_reduced.sample(n=3000, random_state=1)  # يمكنك تعديل العدد حسب الحاجة

# حفظ البيانات الجديدة في ملف CSV
df_reduced.to_csv('tmdb_5000_credits.csv', index=False)

# عرض معلومات حول البيانات الجديدة
print("تم تقليل البيانات وحفظها بنجاح في tmdb_5000_credits_reduced.csv")
print(df_reduced.head())  # عرض أول 5 صفوف من البيانات الجديدة

# تحقق من حجم الملف
print("حجم الملف:", df_reduced.memory_usage(deep=True).sum() / 1024, "KB")
