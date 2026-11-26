import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Données du tableau
night = np.array([10, 11, 12, 13, 14])
real = np.array([1.790, 2.079, 2.310, 2.470, 2.520])
predicted = np.array([1.793, 2.064, 2.318, 2.479, 2.590])

# 2. Calcul des métriques de performance
mse = mean_squared_error(real, predicted)
mae = mean_absolute_error(real, predicted)
r2 = r2_score(real, predicted)
# 3. Calcul de l'Accuracy pour la régression
# On calcule le pourcentage d'erreur absolu moyen (MAPE)
mape = np.mean(np.abs((real - predicted) / real))
# L'accuracy est le complémentaire de cette erreur
accuracy = (1 - mape) * 100
# Affichage des métriques calculées
print("--- Métriques de Performance ---")
print(f"Mean Squared Error (MSE) : {mse:.6f}")
print(f"Mean Absolute Error (MAE) : {mae:.6f}")
print(f"Coefficient de détermination (R²) : {r2:.4f}")
print(f"L'accuracy' : {accuracy}")
# 3. Génération du graphique
plt.figure(figsize=(9, 5))
plt.plot(night, real, marker='o', linestyle='-', color='blue', linewidth=2, label='Réel (Real)')
plt.plot(night, predicted, marker='s', linestyle='--', color='orange', linewidth=2, label='Prédit (Predicted)')

# Configuration des axes et du style
plt.title("Comparaison des valeurs Réelles vs Prédites par Nuit", fontsize=14, fontweight='bold')
plt.xlabel("ID de la Nuit (Night)", fontsize=12)
plt.ylabel("Valeurs", fontsize=12)
plt.xticks(night)  # Forcer l'affichage de chaque ID de nuit
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Affichage du graphique
plt.tight_layout()
plt.show()
