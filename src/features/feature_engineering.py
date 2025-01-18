import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# قراءة بيانات التدريب
train_data = pd.read_csv(r'C:\Users\master save\Downloads\archive (1)\medicalmalpractice.csv')
# قراءة بيانات الاختبار
test_data = pd.read_csv(r'C:\Users\master save\Downloads\archive (1)\medicalmalpractice.csv')

# بناء دالة لمعالجة القيم المفقودة في عمود الحالة الاجتماعية
def handle_missing_values(df):
    # إيجاد القيمة الأكثر تكرارًا لاستبدال القيم المفقودة
    most_frequent_category = df["Marital Status"].mode()[0]
    
    # استبدال القيم المفقودة بالقيمة الأكثر تكرارًا
    df["Marital Status"] = df["Marital Status"].replace(4, most_frequent_category)  # 4 تعني "Unknown" في الحالة الاجتماعية
    return df

# دالة ترميز الهدف للعمود "Specialty"
def target_encoder(df):
    # حساب المتوسط لكل قيمة في العمود "Specialty"
    specialty_means = df.groupby("Specialty")['Amount'].mean()
    # تطبيق المتوسطات على العمود "Specialty"
    df['Specialty'] = df["Specialty"].map(specialty_means)
    return df

# دالة لترميز "Gender" و "Marital Status"
def label_encoder(df):
    # ترميز العمود "Gender"
    df.replace({'Gender': {'Male': 1, 'Female': 0}}, inplace=True)
    # ترميز العمود "Marital Status"
    df.replace({'Marital Status': {'Married': 4, 'Divorced': 3, 'Single': 2, 'Widowed': 1}}, inplace=True)
    return df

# دالة لترميز العمود "Insurance" باستخدام OneHotEncoding
def one_hot_encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[['Insurance']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Insurance']))
    df = pd.concat([df, encoded_df], axis=1)
    # حذف العمود الأصلي "Insurance"
    df.drop('Insurance', axis=1, inplace=True)
    return df

# دالة لتوحيد مقياس القيم
def scaler(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

# تطبيق دالة معالجة القيم المفقودة على بيانات التدريب والاختبار
train_data_processed = handle_missing_values(train_data)
test_data_processed = handle_missing_values(test_data)

# تطبيق دوال الترميز على بيانات التدريب والاختبار
funcs = [target_encoder, label_encoder, one_hot_encoder]
for f in funcs:
    train_data_processed = f(train_data_processed)
    test_data_processed = f(test_data_processed)

# تطبيق توحيد المقياس على بيانات التدريب والاختبار
features_to_scale = ["Severity", "Age", "Marital Status", "Specialty"]
scaler(train_data_processed, features_to_scale)
scaler(test_data_processed, features_to_scale)

# التحقق من القيم الفريدة في عمود الحالة الاجتماعية بعد المعالجة
unique_values = train_data_processed['Marital Status'].unique()
print(f"Unique values in Marital Status after processing: {unique_values}")

# عرض بعض العينات من البيانات المعالجة
print(train_data_processed.head())

# إنشاء مجلد لتخزين البيانات المعالجة
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

# تخزين البيانات المعالجة
train_data_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_data_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
