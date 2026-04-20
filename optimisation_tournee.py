import pandas as pd
import numpy as np
from itertools import permutations # Ajouté pour l'exploration par force brute

# Ces modules sont nécessaires pour le fonctionnement.
# Assurez-vous qu'ils sont accessibles dans votre environnement.
import transformer_horaire
import suggestions_sites # Utilisé pour la fonction calculer_planning_apres_insertion


def optimiser_tournee(sites_df: pd.DataFrame, durations_df: pd.DataFrame, horaire_tech: str) -> pd.DataFrame | None:
    """
    Optimisation de tournée simplifiée par exploration de toutes les permutations possibles
    des sites et sélection de celle qui termine le plus tôt.

    Cette méthode garantit de trouver la meilleure solution (optimum global) mais
    devient très lente avec un nombre élevé de sites (plus de 9-10).

    Args:
        sites_df (pd.DataFrame): DataFrame des sites à visiter, incluant ID_Site, Nom,
                                 Temps_Total_Service, Ouv_Matin, Ferm_Matin, Ouv_Aprem, Ferm_Aprem, Horaires.
        durations_df (pd.DataFrame): DataFrame des durées de trajet brutes entre les sites.
        horaire_tech (str): Chaîne de caractères des horaires du technicien (ex: "08:00-17:30").

    Returns:
        pd.DataFrame: Le DataFrame de l'itinéraire le plus tôt trouvé,
                      avec les colonnes 'ID_Site', 'Lieu', 'Horaires', 'Total Service',
                      'Heure_Debut', 'Heure_Fin', 'Ordre'.
                      Les colonnes 'Heure_Debut' et 'Heure_Fin' sont en minutes depuis minuit.
                      Retourne None si aucun itinéraire faisable n'est trouvé ou si trop de sites.
    """

    if sites_df.empty:
        return None

    # Limite le nombre de sites pour éviter des temps de calcul excessifs avec la force brute.
    # Pour 9 sites, il y a 362 880 permutations. Pour 10, plus de 3,6 millions.
    # Cette limite peut être ajustée en fonction de la performance souhaitée.
    if len(sites_df) > 9:
        print(f"Attention : Le nombre de sites ({len(sites_df)}) est trop élevé pour une optimisation par force brute. "
              "Veuillez sélectionner moins de sites (maximum 9) pour cette méthode simplifiée.")
        return None

    best_tour_df = None
    min_end_time = float('inf') # Initialisé à une valeur très élevée (minutes depuis minuit)

    # Récupère la liste des ID des sites à inclure dans les permutations
    site_ids = sites_df['ID_Site'].tolist()

    # Génère toutes les permutations possibles des ID de sites
    for permutation in permutations(site_ids):
        # Construit un DataFrame temporaire pour l'itinéraire actuel, dans l'ordre de la permutation.
        # Ce DataFrame doit inclure toutes les informations nécessaires à calculer_planning_apres_insertion.
        current_tour_data = []
        for order, site_id in enumerate(permutation):
            site_info = sites_df[sites_df['ID_Site'] == site_id].iloc[0]
            current_tour_data.append({
                "ID_Site": site_id,
                "Ordre": order + 1, # L'ordre est basé sur 1 pour la lisibilité
                "Lieu": site_info['Nom'],
                "Horaires": site_info['Horaires'],
                "Total Service": site_info['Temps_Total_Service'],
                "Ouv_Matin": site_info['Ouv_Matin'],
                "Ferm_Matin": site_info['Ferm_Matin'],
                "Ouv_Aprem": site_info['Ouv_Aprem'],
                "Ferm_Aprem": site_info['Ferm_Aprem'],
            })
        
        df_current_permutation = pd.DataFrame(current_tour_data)

        # Évalue la faisabilité et obtient le planning détaillé pour cette permutation
        # `sites_df` est passé en tant que `df_tous_les_sites` pour que `calculer_planning_apres_insertion`
        # puisse trouver toutes les informations nécessaires des sites.
        planned_tour, is_feasible, _ = suggestions_sites.calculer_planning_apres_insertion(
            df_current_permutation,
            sites_df, # Source de toutes les informations détaillées des sites
            durations_df,
            horaire_tech
        )

        if is_feasible:
            # Récupère l'heure de fin du dernier site dans cet itinéraire planifié.
            # `Heure_Fin` est en minutes depuis minuit, telle que retournée par calculer_planning_apres_insertion.
            if not planned_tour.empty and not planned_tour['Heure_Fin'].isnull().any():
                last_site_end_time_minutes = planned_tour.sort_values(by='Ordre').iloc[-1]['Heure_Fin']

                # Si cet itinéraire termine plus tôt que le meilleur trouvé jusqu'à présent,
                # il devient le nouveau "meilleur itinéraire".
                if last_site_end_time_minutes < min_end_time:
                    min_end_time = last_site_end_time_minutes
                    best_tour_df = planned_tour.copy() # Stocke une copie du meilleur itinéraire

    # Tri final pour s'assurer que l'ordre est correct
    if best_tour_df is not None:
        best_tour_df = best_tour_df.sort_values(by='Ordre').reset_index(drop=True)

    return best_tour_df

# Les fonctions originales liées à OR-Tools (best_itineraire, dataFrame_en_matrice,
# reduire_taille, ajuster_horaire_matin, ajuster_horaire_aprem, appliquer_solveur,
# appliquer_solveur_avec_depot) ne sont plus utilisées par cette nouvelle implémentation
# de `optimiser_tournee` et peuvent être supprimées si elles ne servent pas ailleurs.
