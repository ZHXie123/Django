import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 假设你已经有了一个DataFrame 'df' 包含列 'student_id' 和 'course_id'
# 并且你已经将 'course_student_pairs' 转换为DataFrame
from base import views

course_student_pairs = views.GetForSeeData
# 将数据转换为适合机器学习的格式
df = pd.DataFrame(course_student_pairs, columns=['student_id', 'course_id'])

# 将类别数据编码为数值
label_encoder = LabelEncoder()
df['student_id'] = label_encoder.fit_transform(df['student_id'])
df['course_id'] = label_encoder.fit_transform(df['course_id'])

# 为每个学生创建一个选课矩阵
students = df['student_id'].unique()
courses = df['course_id'].unique()

# 创建一个空的矩阵，用于存储每个学生的选课情况
student_course_matrix = pd.DataFrame(0, index=students, columns=courses)

# 填充矩阵
for index, row in df.iterrows():
    student_course_matrix.at[row['student_id'], row['course_id']] = 1

# 转换为稀疏矩阵以节省内存
student_course_matrix_sparse = student_course_matrix.sparse.to_coo()

# 提取特征和标签
X = student_course_matrix_sparse.toarray()
y = student_course_matrix_sparse.row

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 评估模型
print("模型在测试集上的准确率:", rf.score(X_test, y_test))

# 预测新学生选课情况
# 假设新学生选课情况为一个全0的矩阵
new_student_courses = np.zeros(len(courses))
new_student_courses_encoded = new_student_courses

# 预测
predicted_courses = rf.predict([new_student_courses_encoded])

# 将预测结果转换回原始课程ID
predicted_course_ids = label_encoder.inverse_transform(predicted_courses)

print("新学生可能选择的课程ID:", predicted_course_ids)