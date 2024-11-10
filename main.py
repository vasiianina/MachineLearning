import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Шаг 1: Чтение данных
df = pd.read_csv('/mnt/data/spotify_songs.csv')

# Шаг 2: Визуализация данных и вычисление корреляции
def visualize_data(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица для треков Spotify')
    plt.show()

    df.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.suptitle('Гистограммы признаков треков', y=1.02)
    plt.show()

# Визуализация данных
visualize_data(df)

# Шаг 3: Обработка пропущенных данных
def check_missing_data(df):
    missing_data = df.isnull().sum()
    print("Пропущенные данные:")
    print(missing_data)

check_missing_data(df)

# Шаг 4: Подготовка данных для модели
def prepare_data(df):
    # Выбираем числовые признаки для модели и целевой признак - популярность трека
    X = df.select_dtypes(include=[np.number]).drop('track_popularity', axis=1, errors='ignore')
    y = df['track_popularity'].apply(lambda x: 1 if x >= 50 else 0)  # Бинарная классификация по популярности
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X, y = prepare_data(df)

# Шаг 5: Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 6: Обучение KNN-классификатора
def train_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Точность KNN:", accuracy)
    print("Отчет классификации:")
    print(classification_report(y_test, y_pred))

# Тренировка и оценка KNN
train_knn(X_train, y_train, X_test, y_test)

# Шаг 7: Балансировка классов с использованием SMOTE
def handle_class_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
train_knn(X_train_balanced, y_train_balanced, X_test, y_test)

# Шаг 8: Обучение других классификаторов
def train_other_classifiers(X_train, y_train, X_test, y_test):
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точность для {name}:", accuracy)
        print("Отчет классификации:")
        print(classification_report(y_test, y_pred))

# Тренировка других моделей
train_other_classifiers(X_train_balanced, y_train_balanced, X_test, y_test)

# Шаг 9: Исключение коррелированных признаков
def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"Удаляем следующие коррелированные признаки: {to_drop}")
    return df.drop(to_drop, axis=1)

df_reduced = remove_highly_correlated_features(df)
