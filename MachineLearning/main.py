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
df_white = pd.read_csv('winequality-white.csv', sep=';')
df_red = pd.read_csv('winequality-red.csv', sep=';')

# Шаг 2: Визуализация данных и вычисление корреляции
def visualize_data(df, wine_type):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Корреляционная матрица для {wine_type}')
    plt.show()

    df.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.suptitle(f'Гистограммы для {wine_type}', y=1.02)
    plt.show()

# Визуализация для белого и красного вина
visualize_data(df_white, "белого вина")
visualize_data(df_red, "красного вина")

# Шаг 3: Обработка пропущенных данных (проверка)
def check_missing_data(df, wine_type):
    missing_data = df.isnull().sum()
    print(f"Пропущенные данные для {wine_type}:")
    print(missing_data)

check_missing_data(df_white, "белого вина")
check_missing_data(df_red, "красного вина")

# Шаг 5: Нормализация данных (StandardScaler)
def prepare_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X_white_scaled, y_white = prepare_data(df_white)
X_red_scaled, y_red = prepare_data(df_red)

# Шаг 6: Разбиение на обучающую и тестовую выборки
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_white, X_test_white, y_train_white, y_test_white = split_data(X_white_scaled, y_white)
X_train_red, X_test_red, y_train_red, y_test_red = split_data(X_red_scaled, y_red)

# Шаг 7: Обучение KNN-классификатора
def train_knn(X_train, y_train, X_test, y_test, wine_type):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Результаты для {wine_type} вина:")
    print(f"Точность: {accuracy}")
    print("Отчет классификации:")
    print(class_report)


# Обучение и оценка для белого и красного вина
train_knn(X_train_white, y_train_white, X_test_white, y_test_white, "белого")
train_knn(X_train_red, y_train_red, X_test_red, y_test_red, "красного")


# Шаг 10: Борьба с несбалансированностью классов (SMOTE)
def handle_class_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

X_train_white_balanced, y_train_white_balanced = handle_class_imbalance(X_train_white, y_train_white)
X_train_red_balanced, y_train_red_balanced = handle_class_imbalance(X_train_red, y_train_red)

print("Переподготовка моделей после балансировки данных с использованием SMOTE")
train_knn(X_train_white_balanced, y_train_white_balanced, X_test_white, y_test_white, "белого")
train_knn(X_train_red_balanced, y_train_red_balanced, X_test_red, y_test_red, "красного")

def train_other_classifiers(X_train, y_train, X_test, y_test, wine_type):
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"Результаты для {wine_type} вина с классификатором {name}:")
        print(f"Точность: {accuracy}")
        print("Отчет классификации:")
        print(class_report)


# Обучение других классификаторов
train_other_classifiers(X_train_white_balanced, y_train_white_balanced, X_test_white, y_test_white, "белого")
train_other_classifiers(X_train_red_balanced, y_train_red_balanced, X_test_red, y_test_red, "красного")


# Шаг 11: Исключение коррелированных переменных
def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"Удаляем следующие коррелированные признаки: {to_drop}")
    return df.drop(to_drop, axis=1)


df_white_reduced = remove_highly_correlated_features(df_white)
df_red_reduced = remove_highly_correlated_features(df_red)
