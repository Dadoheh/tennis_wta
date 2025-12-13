# =============================================================================
# Przewidywanie wyników meczów tenisowych WTA
# Algorytm: Random Forest Classifier
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# 1. WCZYTANIE DANYCH
# =============================================================================

# Ścieżka do katalogu ze skryptem
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filepath):
    """Wczytuje dane z pliku CSV"""
    df = pd.read_csv(filepath)
    print(f"Wczytano {len(df)} meczów")
    print(f"Kolumny: {list(df.columns)}")
    return df

# Wczytanie danych z 2024 roku
df = load_data(os.path.join(SCRIPT_DIR, 'wta_matches_2024.csv'))

# Podgląd danych
print("\n=== Podgląd danych ===")
print(df.head())
print(f"\nRozmiar: {df.shape}")

# =============================================================================
# 2. PRZYGOTOWANIE DANYCH (PREPROCESSING)
# =============================================================================

def prepare_match_data(df):
    """
    Przekształca dane z formatu winner/loser na format Player A vs Player B
    z etykietą 1 jeśli Player A wygrał, 0 jeśli przegrał.
    
    Losowo przypisujemy zawodniczkę jako Player A lub B, aby model
    nie uczył się, że Player A zawsze wygrywa.
    """
    matches = []
    
    for idx, row in df.iterrows():
        # Losowo decydujemy, czy winner będzie Player A czy B
        if np.random.random() > 0.5:
            # Winner = Player A
            match = {
                'player_a_ht': row['winner_ht'],
                'player_a_age': row['winner_age'],
                'player_a_hand': row['winner_hand'],
                'player_a_rank': row['winner_rank'],
                'player_a_rank_points': row['winner_rank_points'],
                'player_b_ht': row['loser_ht'],
                'player_b_age': row['loser_age'],
                'player_b_hand': row['loser_hand'],
                'player_b_rank': row['loser_rank'],
                'player_b_rank_points': row['loser_rank_points'],
                'surface': row['surface'],
                'tourney_level': row['tourney_level'],
                'target': 1  # Player A wygrał
            }
        else:
            # Loser = Player A
            match = {
                'player_a_ht': row['loser_ht'],
                'player_a_age': row['loser_age'],
                'player_a_hand': row['loser_hand'],
                'player_a_rank': row['loser_rank'],
                'player_a_rank_points': row['loser_rank_points'],
                'player_b_ht': row['winner_ht'],
                'player_b_age': row['winner_age'],
                'player_b_hand': row['winner_hand'],
                'player_b_rank': row['winner_rank'],
                'player_b_rank_points': row['winner_rank_points'],
                'surface': row['surface'],
                'tourney_level': row['tourney_level'],
                'target': 0  # Player A przegrał
            }
        matches.append(match)
    
    return pd.DataFrame(matches)

# Przekształcenie danych
print("\n=== Przekształcanie danych ===")
np.random.seed(42)  # Dla powtarzalności
match_df = prepare_match_data(df)
print(f"Przekształcono {len(match_df)} meczów")
print(f"\nRozkład wyników: \n{match_df['target'].value_counts()}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    """
    Tworzy dodatkowe cechy (features) na podstawie istniejących danych.
    """
    df = df.copy()
    
    # Różnice między zawodniczkami
    df['height_diff'] = df['player_a_ht'] - df['player_b_ht']
    df['age_diff'] = df['player_a_age'] - df['player_b_age']
    df['rank_diff'] = df['player_a_rank'] - df['player_b_rank']
    df['rank_points_diff'] = df['player_a_rank_points'] - df['player_b_rank_points']
    
    return df

match_df = create_features(match_df)
print("\n=== Nowe cechy ===")
print(match_df[['height_diff', 'age_diff', 'rank_diff', 'rank_points_diff']].describe())

# =============================================================================
# 4. OBSŁUGA BRAKUJĄCYCH WARTOŚCI I ENCODING
# =============================================================================

def preprocess_features(df):
    """
    Obsługuje brakujące wartości i koduje zmienne kategoryczne.
    """
    df = df.copy()
    
    # Kolumny numeryczne - wypełniamy medianą
    numeric_cols = ['player_a_ht', 'player_a_age', 'player_a_rank', 'player_a_rank_points',
                    'player_b_ht', 'player_b_age', 'player_b_rank', 'player_b_rank_points',
                    'height_diff', 'age_diff', 'rank_diff', 'rank_points_diff']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Kolumny kategoryczne - Label Encoding
    le_hand = LabelEncoder()
    le_surface = LabelEncoder()
    le_level = LabelEncoder()
    
    # Wypełniamy brakujące wartości przed enkodowaniem
    df['player_a_hand'] = df['player_a_hand'].fillna('U')  # Unknown
    df['player_b_hand'] = df['player_b_hand'].fillna('U')
    df['surface'] = df['surface'].fillna('Hard')
    df['tourney_level'] = df['tourney_level'].fillna('A')
    
    # Encoding
    df['player_a_hand_encoded'] = le_hand.fit_transform(df['player_a_hand'])
    df['player_b_hand_encoded'] = le_hand.fit_transform(df['player_b_hand'])
    df['surface_encoded'] = le_surface.fit_transform(df['surface'])
    df['tourney_level_encoded'] = le_level.fit_transform(df['tourney_level'])
    
    return df, le_hand, le_surface, le_level

match_df, le_hand, le_surface, le_level = preprocess_features(match_df)
print("\n=== Po preprocessingu ===")
print(f"Brakujące wartości: \n{match_df.isnull().sum()}")

# =============================================================================
# 5. WYBÓR CECH DO MODELU
# =============================================================================

# Cechy do modelu
feature_columns = [
    'player_a_ht', 'player_a_age', 'player_a_rank', 'player_a_rank_points',
    'player_b_ht', 'player_b_age', 'player_b_rank', 'player_b_rank_points',
    'height_diff', 'age_diff', 'rank_diff', 'rank_points_diff',
    'player_a_hand_encoded', 'player_b_hand_encoded',
    'surface_encoded', 'tourney_level_encoded'
]

X = match_df[feature_columns]
y = match_df['target']

print(f"\n=== Dane do modelu ===")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# =============================================================================
# 6. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== Podział danych ===")
print(f"Zbiór treningowy: {X_train.shape[0]} meczów")
print(f"Zbiór testowy: {X_test.shape[0]} meczów")

# =============================================================================
# 7. TRENING MODELU - RANDOM FOREST
# =============================================================================

print("\n=== Trening modelu Random Forest ===")

# Inicjalizacja modelu
rf_model = RandomForestClassifier(
    n_estimators=100,       # Liczba drzew
    max_depth=10,           # Maksymalna głębokość drzewa
    min_samples_split=5,    # Min. próbek do podziału węzła
    min_samples_leaf=2,     # Min. próbek w liściu
    random_state=42,
    n_jobs=-1               # Użyj wszystkich rdzeni CPU
)

# Trening modelu
rf_model.fit(X_train, y_train)
print("Model wytrenowany!")

# =============================================================================
# 8. EWALUACJA MODELU
# =============================================================================

# Predykcje
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Wyniki modelu ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Raport klasyfikacji
print(f"\n=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred, target_names=['Przegrana A', 'Wygrana A']))

# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_names):
    """Wizualizacja ważności cech"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Ważność cech (Feature Importance)')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Cecha')
    plt.ylabel('Ważność')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    
    print("\n=== Ranking ważności cech ===")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

plot_feature_importance(rf_model, feature_columns)

# =============================================================================
# 10. CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_true, y_pred):
    """Wizualizacja macierzy pomyłek"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Przegrana A', 'Wygrana A'],
                yticklabels=['Przegrana A', 'Wygrana A'])
    plt.title('Macierz pomyłek (Confusion Matrix)')
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywista wartość')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# =============================================================================
# 11. PRZYKŁAD PRZEWIDYWANIA
# =============================================================================

def predict_match(model, player_a_stats, player_b_stats, surface, tourney_level):
    """
    Przewiduje wynik meczu między dwoma zawodniczkami.
    
    Parameters:
    -----------
    player_a_stats: dict z kluczami: height, age, hand, rank, rank_points
    player_b_stats: dict z kluczami: height, age, hand, rank, rank_points
    surface: str ('Hard', 'Clay', 'Grass')
    tourney_level: str ('G', 'M', 'P', 'A', 'I', etc.)
    
    Returns:
    --------
    Prawdopodobieństwo wygranej Player A
    """
    # Przygotowanie danych
    features = {
        'player_a_ht': player_a_stats['height'],
        'player_a_age': player_a_stats['age'],
        'player_a_rank': player_a_stats['rank'],
        'player_a_rank_points': player_a_stats['rank_points'],
        'player_b_ht': player_b_stats['height'],
        'player_b_age': player_b_stats['age'],
        'player_b_rank': player_b_stats['rank'],
        'player_b_rank_points': player_b_stats['rank_points'],
        'height_diff': player_a_stats['height'] - player_b_stats['height'],
        'age_diff': player_a_stats['age'] - player_b_stats['age'],
        'rank_diff': player_a_stats['rank'] - player_b_stats['rank'],
        'rank_points_diff': player_a_stats['rank_points'] - player_b_stats['rank_points'],
        'player_a_hand_encoded': le_hand.transform([player_a_stats['hand']])[0],
        'player_b_hand_encoded': le_hand.transform([player_b_stats['hand']])[0],
        'surface_encoded': le_surface.transform([surface])[0],
        'tourney_level_encoded': le_level.transform([tourney_level])[0]
    }
    
    X_pred = pd.DataFrame([features])
    prob = model.predict_proba(X_pred)[0]
    
    return prob[1]  # Prawdopodobieństwo wygranej Player A

# Przykład: Iga Świątek vs Aryna Sabalenka
print("\n=== Przykład przewidywania ===")
print("Mecz: Iga Świątek vs Aryna Sabalenka na Hard (Grand Slam)")

iga_stats = {
    'height': 176,
    'age': 23,
    'hand': 'R',
    'rank': 1,
    'rank_points': 10000
}

sabalenka_stats = {
    'height': 182,
    'age': 26,
    'hand': 'R',
    'rank': 2,
    'rank_points': 8500
}

try:
    prob_iga_wins = predict_match(rf_model, iga_stats, sabalenka_stats, 'Hard', 'G')
    print(f"\nPrawdopodobieństwo wygranej Igi Świątek: {prob_iga_wins:.2%}")
    print(f"Prawdopodobieństwo wygranej Aryny Sabalenki: {1-prob_iga_wins:.2%}")
except Exception as e:
    print(f"Błąd predykcji: {e}")
    print("(Może być spowodowany brakiem niektórych wartości w encoderach)")

print("\n=== Zakończono analizę ===")
