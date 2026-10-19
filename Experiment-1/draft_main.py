import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==========================================
# 1. 加载数据 
# ==========================================
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    # 注意：这里我们直接读取真实的文件名
    train_df = pd.read_csv(BASE_DIR / 'train_data.csv')
    test_df = pd.read_csv(BASE_DIR / 'test_data_unlabeled.csv')
    
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

X_train_raw, y_train_raw, X_test_unlabeled = load_data()
print(f"原始训练集样本数: {len(X_train_raw)}，测试集样本数: {len(X_test_unlabeled)}")

# ==========================================
# 2. 划分训练集与验证集 
# ==========================================
# 重点：先划分数据，再做 TF-IDF，防止数据泄露！
X_tr_text, X_val_text, y_tr, y_val = train_test_split(
    X_train_raw, y_train_raw, 
    test_size=0.2,          # 20% 做验证集
    random_state=42,        # 固定随机种子
    stratify=y_train_raw    # 保证各类别比例一致
)
print(f"划分后 -> 训练集: {len(X_tr_text)}, 验证集: {len(X_val_text)}")

# ==========================================
# 3. TF-IDF 特征提取 (只在训练集 fit)
# ==========================================
vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=50000, 
    sublinear_tf=True
)
# 关键：fit_transform 只用于训练集
X_tr_tfidf = vectorizer.fit_transform(X_tr_text)
# transform 用于验证集和测试集
X_val_tfidf = vectorizer.transform(X_val_text)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)

# ==========================================
# 4. 训练模型并比较 
# ==========================================
models = {
    "朴素贝叶斯": MultinomialNB(alpha=1.0),
    "逻辑回归": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    "线性SVM": LinearSVC(C=1.0, random_state=42)
}

best_model_name = ""
best_f1 = 0
best_model = None

print("\n" + "="*40)
print("开始验证集评估：")
for name, model in models.items():
    model.fit(X_tr_tfidf, y_tr)
    y_pred = model.predict(X_val_tfidf)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"【{name}】 Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print("="*40)
print(f"验证集最佳模型: {best_model_name} (Macro-F1: {best_f1:.4f})")



# ==========================================
# 5. 对测试集进行预测并保存
# ==========================================
# 用最佳模型预测测试集
y_test_pred = best_model.predict(X_test_tfidf)


pred_df = pd.DataFrame({"target": y_test_pred})
pred_df.to_csv('predictions.csv', index=False) 

print("\n预测结果已保存到 predictions.csv")
