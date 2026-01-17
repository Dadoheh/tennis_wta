# Przewidywanie Wyników Meczów Tenisowych WTA
## Dokumentacja Algorytmu Machine Learning

---

## Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Opis algorytmu Random Forest](#opis-algorytmu-random-forest)
3. [Struktura kodu](#struktura-kodu)
4. [Przygotowanie danych](#przygotowanie-danych)
5. [Feature Engineering](#feature-engineering)
6. [Trening i ewaluacja modelu](#trening-i-ewaluacja-modelu)
7. [Jak uruchomić własną analizę](#jak-uruchomić-własną-analizę)
8. [Interpretacja wyników](#interpretacja-wyników)
9. [Silne i słabe strony algorytmu](#silne-i-słabe-strony-algorytmu)
10. [Możliwości rozwoju](#możliwości-rozwoju)

---

## Wprowadzenie

Projekt `AI_alg.py` implementuje model uczenia maszynowego do przewidywania wyników meczów tenisowych WTA (Women's Tennis Association). Model wykorzystuje algorytm **Random Forest Classifier** do klasyfikacji binarnej - przewiduje, która z dwóch zawodniczek wygra mecz na podstawie ich cech (wzrost, wiek, ranking, punkty rankingowe, ręka dominująca) oraz kontekstu meczu (nawierzchnia, poziom turnieju).

### Dane źródłowe

Dane pochodzą z pliku `wta_matches_2024.csv` zawierającego **2689 meczów** z sezonu 2024. Każdy rekord zawiera:
- Informacje o turnieju (nazwa, nawierzchnia, poziom)
- Dane zawodniczek (wzrost, wiek, ranking, punkty)
- Statystyki meczu (asy, podwójne błędy, breakpointy)
- Wynik meczu

---

## Opis algorytmu Random Forest

### Czym jest Random Forest?

**Random Forest** (Las Losowy) to algorytm uczenia maszynowego typu **ensemble** (zespołowy), który łączy wiele drzew decyzyjnych w jeden model. Został opracowany przez Leo Breimana w 2001 roku.

### Zasada działania

```
                    ┌─────────────────┐
                    │   Dane wejściowe│
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Drzewo 1    │ │   Drzewo 2    │ │   Drzewo N    │
    │  (próbka 1)   │ │  (próbka 2)   │ │  (próbka N)   │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │  Predykcja 1  │ │  Predykcja 2  │ │  Predykcja N  │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GŁOSOWANIE    │
                    │  (Voting/Mean)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Finalna decyzja │
                    └─────────────────┘
```

### Kluczowe koncepcje

#### 1. Bootstrap Aggregating (Bagging)
Każde drzewo jest trenowane na losowej próbce danych (z powtórzeniami). Dzięki temu każde drzewo "widzi" nieco inne dane.

#### 2. Losowy wybór cech
Przy każdym podziale węzła, algorytm rozważa tylko losowy podzbiór cech (zwykle √n dla klasyfikacji). Zapobiega to korelacji między drzewami.

#### 3. Głosowanie większościowe
Dla klasyfikacji, każde drzewo oddaje "głos" na jedną z klas. Klasa z największą liczbą głosów wygrywa.

### Matematyczny opis

Dla klasyfikacji, predykcja Random Forest dla obserwacji $x$ to:

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_B(x)\}$$

gdzie:
- $h_B(x)$ - predykcja B-tego drzewa
- $B$ - liczba drzew w lesie
- $\text{mode}$ - wartość najczęściej występująca (moda)

Prawdopodobieństwo przynależności do klasy $c$:

$$P(y = c | x) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}[h_B(x) = c]$$

---

## Struktura kodu

### Diagram przepływu danych

```
┌─────────────────────────────────────────────────────────────┐
│                    AI_alg.py - Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. WCZYTANIE DANYCH                                        │
│     └── load_data() → DataFrame (2689 meczów)               │
│                                                             │
│  2. PRZEKSZTAŁCENIE DANYCH                                  │
│     └── prepare_match_data() → Player A vs Player B         │
│         (losowe przypisanie winner/loser)                   │
│                                                             │
│  3. FEATURE ENGINEERING                                     │
│     └── create_features() → różnice cech                    │
│         (height_diff, age_diff, rank_diff, rank_points_diff)│
│                                                             │
│  4. PREPROCESSING                                           │
│     └── preprocess_features()                               │
│         ├── Uzupełnienie brakujących wartości (mediana)     │
│         └── Label Encoding (hand, surface, tourney_level)   │
│                                                             │
│  5. PODZIAŁ DANYCH                                          │
│     └── train_test_split (80% train / 20% test)             │
│                                                             │
│  6. TRENING MODELU                                          │
│     └── RandomForestClassifier.fit()                        │
│                                                             │
│  7. EWALUACJA                                               │
│     ├── accuracy_score                                      │
│     ├── classification_report                               │
│     ├── confusion_matrix                                    │
│     └── feature_importances_                                │
│                                                             │
│  8. PREDYKCJA                                               │
│     └── predict_match() → prawdopodobieństwo wygranej       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Moduły i funkcje

| Funkcja | Opis | Wejście | Wyjście |
|---------|------|---------|---------|
| `load_data(filepath)` | Wczytuje dane CSV | ścieżka do pliku | DataFrame |
| `prepare_match_data(df)` | Przekształca dane do formatu A vs B | DataFrame | DataFrame |
| `create_features(df)` | Tworzy cechy różnicowe | DataFrame | DataFrame |
| `preprocess_features(df)` | Obsługuje braki, koduje zmienne | DataFrame | DataFrame + Encoders |
| `plot_feature_importance()` | Wizualizuje ważność cech | Model, nazwy cech | Wykres PNG |
| `plot_confusion_matrix()` | Wizualizuje macierz pomyłek | y_true, y_pred | Wykres PNG |
| `predict_match()` | Przewiduje wynik meczu | statystyki zawodniczek | prawdopodobieństwo |

---

## Przygotowanie danych

### Problem z oryginalnym formatem danych

Oryginalne dane zawierają kolumny `winner_*` i `loser_*`, co oznacza, że model mógłby "oszukiwać" - jeśli nauczy się, że winner zawsze wygrywa, będzie miał 100% accuracy, ale nie będzie użyteczny.

### Rozwiązanie: Losowe przypisanie

```python
for idx, row in df.iterrows():
    if np.random.random() > 0.5:
        # Winner = Player A → target = 1
    else:
        # Loser = Player A → target = 0
```

**Efekt:** Zbalansowany zbiór danych z ~50% klas 0 i 1.

### Obsługa brakujących wartości

| Typ zmiennej | Strategia | Uzasadnienie |
|--------------|-----------|--------------|
| Numeryczne (wzrost, wiek, ranking) | Mediana | Odporna na wartości odstające |
| Kategoryczne (ręka) | 'U' (Unknown) | Zachowuje informację o braku danych |
| Kategoryczne (nawierzchnia) | 'Hard' | Najczęstsza nawierzchnia |

### Label Encoding

Zmienne kategoryczne są kodowane numerycznie:

| Zmienna | Wartości | Encoding |
|---------|----------|----------|
| `hand` | R, L, U | 0, 1, 2 |
| `surface` | Hard, Clay, Grass, Carpet | 0, 1, 2, 3 |
| `tourney_level` | G, M, P, A, I, ... | 0, 1, 2, ... |

---

## Feature Engineering

### Utworzone cechy różnicowe

Model wykorzystuje nie tylko surowe wartości, ale także **różnice między zawodniczkami**:

| Cecha | Wzór | Interpretacja |
|-------|------|---------------|
| `height_diff` | $h_A - h_B$ | Przewaga wzrostu Player A |
| `age_diff` | $age_A - age_B$ | Różnica wieku |
| `rank_diff` | $rank_A - rank_B$ | Różnica rankingów (ujemna = A lepsza) |
| `rank_points_diff` | $pts_A - pts_B$ | Różnica punktów (dodatnia = A lepsza) |

### Wszystkie cechy modelu (16)

```python
feature_columns = [
    # Cechy Player A
    'player_a_ht', 'player_a_age', 'player_a_rank', 'player_a_rank_points',
    # Cechy Player B
    'player_b_ht', 'player_b_age', 'player_b_rank', 'player_b_rank_points',
    # Cechy różnicowe
    'height_diff', 'age_diff', 'rank_diff', 'rank_points_diff',
    # Cechy zakodowane
    'player_a_hand_encoded', 'player_b_hand_encoded',
    'surface_encoded', 'tourney_level_encoded'
]
```

---

## Trening i ewaluacja modelu

### Hiperparametry Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100,       # Liczba drzew w lesie
    max_depth=10,           # Maksymalna głębokość każdego drzewa
    min_samples_split=5,    # Min. próbek wymaganych do podziału węzła
    min_samples_leaf=2,     # Min. próbek w liściu
    random_state=42,        # Ziarno losowości (powtarzalność)
    n_jobs=-1               # Użyj wszystkich rdzeni CPU
)
```

### Znaczenie hiperparametrów

| Parametr | Wartość | Wpływ |
|----------|---------|-------|
| `n_estimators` | 100 | Więcej drzew = stabilniejszy model, dłuższy trening |
| `max_depth` | 10 | Ogranicza złożoność, zapobiega overfittingowi |
| `min_samples_split` | 5 | Zapobiega tworzeniu zbyt małych węzłów |
| `min_samples_leaf` | 2 | Zapobiega tworzeniu "jednoelementowych" liści |

### Metryki ewaluacji

#### 1. Accuracy (Dokładność)
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Wynik:** 60.04%

#### 2. Precision (Precyzja)
$$Precision = \frac{TP}{TP + FP}$$

Ile z przewidzianych wygranych faktycznie wygrało?

#### 3. Recall (Czułość)
$$Recall = \frac{TP}{TP + FN}$$

Ile faktycznych wygranych zostało poprawnie przewidzianych?

#### 4. F1-Score
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

Średnia harmoniczna precyzji i czułości.

### Feature Importance

Random Forest pozwala ocenić, które cechy są najważniejsze dla predykcji:

| Rank | Cecha | Ważność |
|------|-------|---------|
| 1 | `rank_points_diff` | 14.75% |
| 2 | `rank_diff` | 12.66% |
| 3 | `player_a_rank_points` | 9.24% |
| 4 | `player_b_rank_points` | 8.47% |
| 5 | `player_a_rank` | 8.26% |

**Wniosek:** Ranking i punkty rankingowe są najlepszymi predyktorami wyniku meczu.

---

## Jak uruchomić własną analizę

### Wymagania

```bash
# Instalacja bibliotek
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Uruchomienie skryptu

```bash
# Windows (PowerShell)
python AI_alg.py

# Lub z konkretnym środowiskiem wirtualnym
& "venv/Scripts/python.exe" AI_alg.py
```

### Własna predykcja meczu

Aby przewidzieć wynik meczu, zmodyfikuj sekcję na końcu `AI_alg.py`:

```python
# Statystyki Player A
player_a_stats = {
    'height': 176,      # Wzrost w cm
    'age': 23,          # Wiek
    'hand': 'R',        # R = praworęczna, L = leworęczna, U = nieznana
    'rank': 1,          # Pozycja w rankingu WTA
    'rank_points': 10000  # Punkty rankingowe
}

# Statystyki Player B
player_b_stats = {
    'height': 182,
    'age': 26,
    'hand': 'R',
    'rank': 2,
    'rank_points': 8500
}

# Kontekst meczu
surface = 'Hard'      # 'Hard', 'Clay', 'Grass', 'Carpet'
tourney_level = 'G'   # 'G'=Grand Slam, 'M'=Masters, 'P'=Premier, 'A'=250

# Wywołanie predykcji
prob = predict_match(rf_model, player_a_stats, player_b_stats, surface, tourney_level)
print(f"Prawdopodobieństwo wygranej Player A: {prob:.2%}")
```

### Kody poziomów turniejów

| Kod | Poziom turnieju |
|-----|-----------------|
| G | Grand Slam (Australian Open, Roland Garros, Wimbledon, US Open) |
| M | WTA 1000 (Masters) |
| P | WTA 500 (Premier) |
| A | WTA 250 |
| I | International / United Cup |

---

## Interpretacja wyników

### Accuracy 60% - czy to dobrze?

**Tak, dla tego problemu to przyzwoity wynik:**

1. **Baseline:** Losowe zgadywanie daje 50% accuracy
2. **Nieprzewidywalność sportu:** Tenis ma wiele niespodzianek
3. **Ograniczone cechy:** Używamy tylko podstawowych cech, bez historii meczów

### Confusion Matrix

```
                           Predykcja
                      Przegrana  Wygrana
Rzeczywista  Przegrana   147      120
             Wygrana      95      176
```

- **True Positives (TP):** 176 - poprawnie przewidziane wygrane A
- **True Negatives (TN):** 147 - poprawnie przewidziane przegrane A
- **False Positives (FP):** 120 - błędnie przewidziane wygrane A
- **False Negatives (FN):** 95 - błędnie przewidziane przegrane A

---

## Silne i słabe strony algorytmu

### Zalety Random Forest

| Zaleta | Opis |
|--------|------|
| **Odporność na overfitting** | Agregacja wielu drzew zmniejsza wariancję |
| **Obsługa różnych typów danych** | Radzi sobie z numerycznymi i kategorycznymi |
| **Brak konieczności skalowania** | Nie wymaga normalizacji danych |
| **Feature Importance** | Pokazuje, które cechy są najważniejsze |
| **Obsługa brakujących wartości** | Relatywnie odporna na braki w danych |
| **Równoległe przetwarzanie** | Drzewa można trenować równolegle |
| **Stabilność** | Małe zmiany w danych nie zmieniają drastycznie wyników |

### Wady Random Forest

| Wada | Opis |
|------|------|
| **Czarna skrzynka** | Trudniej zinterpretować niż pojedyncze drzewo |
| **Pamięciożerność** | Przechowuje wiele drzew w pamięci |
| **Wolniejsza predykcja** | Musi przejść przez wszystkie drzewa |
| **Nie ekstrapoluje** | Nie radzi sobie z wartościami spoza zakresu treningowego |
| **Bias przy niezbalansowanych klasach** | Może faworyzować klasę większościową |

### Porównanie z innymi algorytmami

| Algorytm | Accuracy* | Interpretowalność | Szybkość | Złożoność |
|----------|-----------|-------------------|----------|-----------|
| **Random Forest** | 60% | Średnia | Średnia | Średnia |
| Logistic Regression | ~55% | Wysoka | Wysoka | Niska |
| Decision Tree | ~55% | Wysoka | Wysoka | Niska |
| XGBoost | ~62% | Niska | Średnia | Wysoka |
| Neural Network | ~63% | Bardzo niska | Niska | Bardzo wysoka |

*Szacowane wartości dla tego zbioru danych

---

## Możliwości rozwoju

### 1. Więcej danych

```python
# Wczytanie danych z wielu lat
import glob
files = glob.glob('wta_matches_20*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
```

### 2. Dodatkowe cechy (Feature Engineering)

- **Head-to-head:** Historia bezpośrednich spotkań
- **Forma:** Win-rate z ostatnich 10 meczów
- **Specjalizacja nawierzchniowa:** Win-rate na danej nawierzchni
- **Clutch performance:** Skuteczność w tie-breakach

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Najlepsze parametry: {grid_search.best_params_}")
```

### 4. Inne algorytmy

```python
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()

# XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier()

# Neural Network
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32))
```

### 5. Walidacja krzyżowa

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

---

## Bibliografia

1. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning". Springer.
3. Scikit-learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html#random-forests

---

## Struktura projektu

```
tennis_wta/
├── AI_alg.py              # Główny skrypt z algorytmem
├── README_alg.md          # Ta dokumentacja
├── wta_matches_2024.csv   # Dane meczów
├── feature_importance.png # Wykres ważności cech (generowany)
├── confusion_matrix.png   # Macierz pomyłek (generowana)
└── venv/                  # Środowisko wirtualne Python
```

---

Wykresy:

![Wykres1](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/confusion_matrix.png "Wykres1")
![Wykres2](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/feature_importance.png "Wykres2")
![Wykres3](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/players_age_distribution.png "Wykres3")
![Wykres4](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/players_hand_distribution.png "Wykres4")
![Wykres5](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/players_height_distribution.png "Wykres5")
![Wykres6](https://github.com/Dadoheh/tennis_wta/blob/c4b4720491cf26ef5c11df4e27d51fe811f2edb2/surface_match_distribution.png "Wykres6")

