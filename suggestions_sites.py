import pandas as pd
import numpy as np
import math

# --- Modules externes ---
import transformer_horaire

# --- 0. Constantes et configurations globales ---
# Seuil de durée de trajet en minutes pour considérer les sites comme "proches"
SEUIL_PROXIMITE_MINUTES = 60 # 1 heure
# Rayon minimum en km pour la vérification de point dans un cercle,
# évite les rayons nuls ou très petits qui peuvent causer des problèmes numériques.
RAYON_MINIMUM_KM = 5


# --- 1. Fonctions de Pré-traitement ---
def preparer_matrice_durees(df_durees_brut: pd.DataFrame) -> np.ndarray:
    """
    Convertit le DataFrame brut des durées de trajet en une matrice NumPy
    numérique (float) pour des accès optimisés. Gère les valeurs manquantes
    en les remplaçant par zéro.

    Args:
        df_durees_brut (pd.DataFrame): DataFrame initial contenant les durées
                                       de trajet entre les sites. Attendu avec
                                       une colonne 'id' (qui sera ignorée) et
                                       des colonnes représentant les IDs des sites.
                                       Les données de durée peuvent être des strings
                                       vides ou des nombres.

    Returns:
        np.ndarray: Matrice NumPy (float) des durées de trajet en minutes.
                    Les index (i, j) correspondent aux sites (i+1) et (j+1)
                    si les IDs de sites sont 1-basés.
    """
    # Crée une copie pour éviter de modifier le DataFrame original si cela n'est pas voulu
    df_temp = df_durees_brut.copy()

    # Supprime la colonne 'id' et convertit le reste en matrice NumPy
    # Les valeurs non numériques ou vides sont remplacées par 0.0 et le type est forcé à float.
    matrice_durees = df_temp.drop('id', axis=1, errors='ignore').to_numpy()
    
    # Remplacer les chaînes vides par '0' avant la conversion en float
    # Utilise np.vectorize pour appliquer une fonction sur chaque élément de la matrice
    to_float_or_zero = np.vectorize(lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().replace('.', '', 1).isdigit()) else 0.0)
    matrice_durees = to_float_or_zero(matrice_durees)
    
    return matrice_durees


def parser_horaires_technicien(chaine_horaires_technicien: str) -> tuple[int, int]:
    """
    Convertit une chaîne de caractères représentant les horaires d'un technicien
    (ex: "08:00-17:30") en un tuple d'entiers (heures en minutes depuis minuit).

    Args:
        chaine_horaires_technicien (str): Chaîne de caractères décrivant les
                                         horaires du technicien.

    Returns:
        tuple[int, int]: Un tuple contenant (heure_debut_minutes, heure_fin_minutes).
                         heure_debut_minutes est le nombre de minutes depuis minuit
                         pour le début du shift, et de même pour heure_fin_minutes.
    """

    return transformer_horaire.parser_plage_horaire(chaine_horaires_technicien)


# --- 2. Fonctions Utilitaires de Géométrie ---
def calculer_distance_gps_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """
    Calcule la distance euclidienne approximative entre deux points GPS
    (longitude, latitude) en kilomètres.

    Args:
        point_a (tuple[float, float]): Coordonnées du premier point (longitude, latitude).
        point_b (tuple[float, float]): Coordonnées du deuxième point (longitude, latitude).

    Returns:
        float: Distance euclidienne entre les deux points en kilomètres.
    """
    lon1, lat1 = point_a
    lon2, lat2 = point_b

    # Approximation pour 1 degré de latitude (environ 111.132 km)
    delta_lat = (lat2 - lat1) * 111.132
    
    # Calcul de la latitude moyenne pour ajuster la longueur d'un degré de longitude
    moy_lat_rad = math.radians((lat1 + lat2) / 2)
    # Approximation pour 1 degré de longitude (environ 111.320 km à l'équateur) ajustée par cos(latitude)
    delta_long = (lon2 - lon1) * 111.320 * math.cos(moy_lat_rad)

    return math.sqrt(delta_lat**2 + delta_long**2)


def trouver_milieu_segment_et_longueur(lat_depart: float, long_depart: float,
                                        lat_arrivee: float, long_arrivee: float) -> tuple[tuple[float, float], float]:
    """
    Calcule les coordonnées GPS du point milieu d'un segment et la longueur
    de ce segment entre deux points donnés.

    Args:
        lat_depart (float): Latitude du point de départ.
        long_depart (float): Longitude du point de départ.
        lat_arrivee (float): Latitude du point d'arrivée.
        long_arrivee (float): Longitude du point d'arrivée.

    Returns:
        tuple[tuple[float, float], float]: Un tuple contenant :
            - Un tuple (longitude_milieu, latitude_milieu) pour le point milieu.
            - Un float pour la longueur du segment en kilomètres.
    """
    longueur = round(calculer_distance_gps_km((long_depart, lat_depart), (long_arrivee, lat_arrivee)), 2)
    lat_milieu = round((lat_arrivee + lat_depart) / 2, 7)
    long_milieu = round((long_arrivee + long_depart) / 2, 7)
    
    return (long_milieu, lat_milieu), longueur


def est_point_dans_cercle(centre_cercle_gps: tuple[float, float], rayon_cercle_km: float,
                          point_a_verifier_gps: tuple[float, float]) -> bool:
    """
    Vérifie si un point GPS donné se trouve à l'intérieur ou sur la circonférence
    d'un cercle défini par un centre et un rayon.

    Args:
        centre_cercle_gps (tuple[float, float]): Coordonnées (longitude, latitude)
                                                 du centre du cercle.
        rayon_cercle_km (float): Rayon du cercle en kilomètres.
        point_a_verifier_gps (tuple[float, float]): Coordonnées (longitude, latitude)
                                                    du point à vérifier.

    Returns:
        bool: True si le point est dans le cercle, False sinon.
    """
    # S'assurer que le rayon est au minimum RAYON_MINIMUM_KM pour éviter des rayons trop petits
    rayon_effectif = max(rayon_cercle_km, RAYON_MINIMUM_KM)
    return calculer_distance_gps_km(centre_cercle_gps, point_a_verifier_gps) < rayon_effectif


# --- 3. Fonctions de Calcul des Temps et Contraintes ---

def calculer_temps_trajet_supplementaire(id_site_precedent_insertion: int, id_site_suivant_insertion: int,
                                          id_site_a_inserer: int, matrice_durees: np.ndarray) -> float:
    """
    Calcule le temps de trajet supplémentaire (en minutes) induit par l'insertion
    d'un nouveau site dans un itinéraire. Gère les insertions au début (0, site_suivant),
    à la fin (site_precedent, 0) ou au milieu (site_precedent, site_suivant).

    Args:
        id_site_precedent_insertion (int): ID du site précédent le point d'insertion.
                                           Utiliser 0 si l'insertion est au début de l'itinéraire.
        id_site_suivant_insertion (int): ID du site suivant le point d'insertion.
                                         Utiliser 0 si l'insertion est à la fin de l'itinéraire.
        id_site_a_inserer (int): ID du site que l'on souhaite insérer.
        matrice_durees (np.ndarray): Matrice des durées de trajet en minutes.
                                     Les indices sont 0-basés, donc (ID_site - 1).

    Returns:
        float: Le temps de trajet supplémentaire en minutes.
    """
    # Les ID de sites sont 1-basés, donc pour accéder à la matrice NumPy (0-basée),
    # il faut soustraire 1.

    if id_site_precedent_insertion == 0 and id_site_suivant_insertion == 0:
        return 0.0 # Itinéraire vide, pas de trajet supplémentaire initial

    if id_site_precedent_insertion == 0: # Insertion au début de l'itinéraire
        # Trajet : du site à insérer vers le site suivant
        return matrice_durees[id_site_a_inserer - 1, id_site_suivant_insertion - 1]
    
    if id_site_suivant_insertion == 0: # Insertion à la fin de l'itinéraire
        # Trajet : du site précédent vers le site à insérer
        return matrice_durees[id_site_precedent_insertion - 1, id_site_a_inserer - 1]
    
    # Insertion au milieu de l'itinéraire
    # Calcul du trajet si le nouveau site est inséré
    trajet_via_nouveau = (matrice_durees[id_site_precedent_insertion - 1, id_site_a_inserer - 1] +
                          matrice_durees[id_site_a_inserer - 1, id_site_suivant_insertion - 1])
    
    # Trajet existant entre les deux sites
    trajet_existant = matrice_durees[id_site_precedent_insertion - 1, id_site_suivant_insertion - 1]
    
    return round(trajet_via_nouveau - trajet_existant, 2)


def trouver_sites_ouverture_tot_proches(id_site_reference: int, matrice_durees: np.ndarray,
                                         df_horaires_sites: pd.DataFrame) -> list[int]:
    """
    Identifie les IDs des sites qui ouvrent tôt le matin (avant 8h30, 510 minutes depuis minuit)
    et qui sont à moins d'un certain seuil de temps de trajet du site de référence.

    Args:
        id_site_reference (int): ID du site de référence (site autour duquel on cherche).
        matrice_durees (np.ndarray): Matrice des durées de trajet en minutes.
        df_horaires_sites (pd.DataFrame): DataFrame des informations sur les sites,
                                          incluant 'ID_Site' et 'Ouv_Matin'.

    Returns:
        list[int]: Liste des IDs des sites correspondants.
    """
    # Filtrer les sites qui ouvrent tôt le matin (ex: avant 8h30, soit 510 minutes depuis minuit)
    # Assurez-vous que 'Ouv_Matin' est bien en minutes depuis minuit.
    ids_sites_ouvrant_tot = df_horaires_sites[df_horaires_sites['Ouv_Matin'] <= 510]['ID_Site'].tolist()
    sites_proches = []
    for id_site_candidat in ids_sites_ouvrant_tot:
        # Éviter de suggérer le site de référence lui-même
        if id_site_candidat == id_site_reference:
            continue
        
        # Vérifier si la durée de trajet du site candidat vers le site de référence est inférieure au seuil
        # (C'est un trajet "aller", pour savoir si on peut arriver là-bas tôt depuis un autre endroit)
        if matrice_durees[id_site_candidat - 1, id_site_reference - 1] < SEUIL_PROXIMITE_MINUTES:
            sites_proches.append(id_site_candidat)
    
    return sites_proches


def trouver_sites_fermeture_tard_proches(id_site_reference: int, matrice_durees: np.ndarray,
                                           df_horaires_sites: pd.DataFrame) -> list[int]:
    """
    Identifie les IDs des sites qui ferment tard le soir (après 16h30, 990 minutes depuis minuit)
    et qui sont accessibles en moins d'un certain seuil de temps de trajet
    depuis le site de référence.

    Args:
        id_site_reference (int): ID du site de référence (site depuis lequel on cherche).
        matrice_durees (np.ndarray): Matrice des durées de trajet en minutes.
        df_horaires_sites (pd.DataFrame): DataFrame des informations sur les sites,
                                          incluant 'ID_Site' et une colonne appropriée
                                          pour la fermeture du soir (ex: 'Ferm_Aprem').

    Returns:
        list[int]: Liste des IDs des sites correspondants.
    """
    # Filtrer les sites qui ferment tard le soir (ex: après 16h30, soit 990 minutes depuis minuit)
    # ATTENTION: Utiliser la colonne correcte pour la fermeture. 'Ferm_Aprem' est un bon candidat
    # si elle contient les minutes depuis minuit pour la fermeture de l'après-midi.
    ids_sites_fermant_tard = df_horaires_sites[(df_horaires_sites['Ferm_Aprem'] >= 990) | (df_horaires_sites['Ferm_Matin'] >= 990)]['ID_Site'].tolist()

    sites_proches = []
    for id_site_candidat in ids_sites_fermant_tard:
        if id_site_candidat == id_site_reference:
            continue
        
        # Vérifier si la durée de trajet du site de référence vers le site candidat est inférieure au seuil
        # (C'est un trajet "retour" ou "extension", pour savoir si on peut y aller après)
        if matrice_durees[id_site_reference - 1, id_site_candidat - 1] < SEUIL_PROXIMITE_MINUTES:
            sites_proches.append(id_site_candidat)
    
    return sites_proches


def calculer_planning_apres_insertion(df_itineraire_a_planifier: pd.DataFrame, df_tous_les_sites: pd.DataFrame,
                                     df_durees_brut: pd.DataFrame, chaine_horaires_technicien: str) -> tuple[pd.DataFrame, bool]:
    """
    Calcule le planning détaillé (Heure_Debut, Heure_Fin) pour chaque site d'un itinéraire
    et vérifie sa faisabilité en respectant les contraintes d'horaires des sites et du technicien.
    Cette fonction est une version étendue de 'verifier_faisabilite_itineraire' qui remplit
    également les heures de début et de fin.

    Args:
        df_itineraire_a_planifier (pd.DataFrame): DataFrame de l'itinéraire, doit contenir
                                                  'ID_Site', 'Ordre', 'Total Service' et les
                                                  colonnes d'horaires des sites (Ouv_Matin, etc.).
                                                  Une COPIE sera faite pour éviter la modification de l'original.
        df_tous_les_sites (pd.DataFrame): Informations détaillées sur tous les sites,
                                          incluant 'ID_Site', 'Nom', 'Ouv_Matin', 'Ferm_Matin',
                                          'Ouv_Aprem', 'Ferm_Aprem'. (Utilisé pour récupérer les détails des sites si manquants dans df_itineraire_a_planifier).
        df_durees_brut (pd.DataFrame): DataFrame original des durées (brut).
        chaine_horaires_technicien (str): Chaîne des horaires du technicien.

    Returns:
        tuple[pd.DataFrame, bool]: Un tuple contenant :
            - pd.DataFrame: L'itinéraire planifié avec 'Heure_Debut' et 'Heure_Fin' remplis.
                           Si l'itinéraire est infaisable, les heures peuvent être partielles
                           ou non remplies après le point d'échec.
            - bool: True si l'itinéraire est entièrement faisable, False sinon.
    """
    # Pré-traitement des données
    matrice_durees = preparer_matrice_durees(df_durees_brut)
    horaires_technicien_minutes = parser_horaires_technicien(chaine_horaires_technicien)

    temps_perdu = 0 

    df_itineraire_planifie = df_itineraire_a_planifier.copy() # Travailler sur une copie
    
    # Assurer que df_itineraire_planifie a toutes les colonnes de détails des sites nécessaires
    # (Ouv_Matin, Ferm_Matin, etc.) en fusionnant avec df_tous_les_sites si elles manquent.
    colonnes_details_sites = ['ID_Site', 'Ouv_Matin', 'Ferm_Matin', 'Ouv_Aprem', 'Ferm_Aprem']
    for col in colonnes_details_sites:
        if col not in df_itineraire_planifie.columns:
            df_itineraire_planifie = pd.merge(
                df_itineraire_planifie, 
                df_tous_les_sites[colonnes_details_sites], 
                on='ID_Site', 
                how='left'
            )
            break # Une fois fusionné, on a toutes les colonnes

    # Indicateur pour savoir si la pause déjeuner a déjà été prise
    flag_pause = 1 # 1: pas encore fait la pause, 0: pause faite
    
    # Horaires de début et fin de journée du technicien en minutes depuis minuit
    debut_tech, fin_tech = horaires_technicien_minutes
    
    # Trier l'itinéraire par ordre pour une vérification séquentielle
    df_itineraire_planifie = df_itineraire_planifie.sort_values(by='Ordre').reset_index(drop=True)
    
    # Initialiser le temps actuel du technicien à son heure de début
    temps_courant = float(debut_tech)
    id_site_prec = 0 # ID du site précédent, 0 pour le point de départ virtuel
    
    # Calculer le temps de travail disponible total (incluant la pause)
    temps_travail_total_brut = fin_tech - debut_tech
    temps_pause_a_deduire = 90 if temps_travail_total_brut > (4 * 60) else 0 # 90 min de pause si journée > 4h
    temps_travail_disponible_net = temps_travail_total_brut - temps_pause_a_deduire
    
    # Vérifier si le temps total de service dépasse le temps de travail disponible
    somme_temps_service = df_itineraire_planifie['Total Service'].sum()
    if somme_temps_service > temps_travail_disponible_net:
        return df_itineraire_planifie, False,temps_perdu # Infaisable
    
    #temps perdu au moment de la pause de midi 
    

    for idx, row in df_itineraire_planifie.iterrows():
        id_site_actuel = row['ID_Site']

        # Les IDs de site 0 sont considérés comme des points de dépôt/virtuels et ne sont pas vérifiés
        if id_site_actuel == 0 or id_site_actuel > 1000: # Si 0 est inclus dans l'itinéraire pour une raison quelconque
            df_itineraire_planifie.loc[idx, 'Heure_Debut'] = np.nan
            df_itineraire_planifie.loc[idx, 'Heure_Fin'] = np.nan
            continue

        # Récupérer le temps de trajet depuis le site précédent
        if id_site_prec == 0:
            temps_trajet = 0.0 # Pas de trajet si c'est le premier site
        else:
            # Assurez-vous que l'indexation est correcte et que id_site_prec - 1 et id_site_actuel - 1 sont valides
            try:
                temps_trajet = matrice_durees[id_site_prec - 1, id_site_actuel - 1]
            except IndexError:
                return df_itineraire_planifie, False,temps_perdu # Itinéraire infaisable

        temps_courant += temps_trajet # Le technicien arrive sur le site

        # Récupérer les informations horaires du site actuel (elles sont censées être dans df_itineraire_planifie)
        service_time = row['Total Service']
        ouv_matin = row['Ouv_Matin']
        ferm_matin = row['Ferm_Matin']
        ouv_aprem = row['Ouv_Aprem']
        ferm_aprem = row['Ferm_Aprem']

        # Gérer l'arrivée sur le site et l'attente éventuelle
        if temps_courant < ouv_matin:
            temps_courant = float(ouv_matin) # Le technicien attend l'ouverture du matin
        
        heure_debut_travail_site = temps_courant

        # Vérifier si le service peut être effectué entièrement le matin
        if (heure_debut_travail_site + service_time) <= ferm_matin:
            temps_courant += float(service_time) # Le service est fait le matin

        elif (ouv_aprem <=0):
            return df_itineraire_planifie, False,temps_perdu
        else:

            heure_debut_travail_site = temps_courant # Mise à jour de l'heure de l'arrivée sur le lieu de travail
            # Le service ne peut pas être fini le matin ou le technicien arrive après la fermeture du matin.
            # Gérer la pause déjeuner.
            if flag_pause == 1 and temps_pause_a_deduire > 0: # Si la pause n'a pas été prise et est applicable
                temps_courant += float(temps_pause_a_deduire)
                flag_pause = 0 # Marquer la pause comme prise
            
            # Après la pause (ou si déjà après la matinée), vérifier l'ouverture de l'après-midi
            if temps_courant < ouv_aprem:
                temps_perdu += ouv_aprem - temps_courant
                temps_courant = float(ouv_aprem) # Le technicien attend l'ouverture de l'après-midi
            
            

            # Vérifier si le service peut être effectué l'après-midi
            if (heure_debut_travail_site + service_time) > ferm_aprem:
                # Si infaisable, marquer les heures comme NaN pour ce site et les suivants
                df_itineraire_planifie.loc[idx:, ['Heure_Debut', 'Heure_Fin']] = np.nan
                return df_itineraire_planifie, False,temps_perdu # Infaisable

            temps_courant += float(service_time) # Le service est fait l'après-midi

        # Vérifier si le technicien termine le site après ses propres horaires de fin de journée
        if temps_courant > fin_tech:
            df_itineraire_planifie.loc[idx:, ['Heure_Debut', 'Heure_Fin']] = np.nan
            return df_itineraire_planifie, False,temps_perdu # Infaisable

        # Remplir les heures de début et de fin pour ce site
        df_itineraire_planifie.loc[idx, 'Heure_Debut'] = int(heure_debut_travail_site)
        df_itineraire_planifie.loc[idx, 'Heure_Fin'] = int(temps_courant)
        
        id_site_prec = id_site_actuel # Mettre à jour le site précédent pour la prochaine itération

    return df_itineraire_planifie, True,temps_perdu # Si toutes les contraintes sont respectées


# --- 4. Fonctions de Manipulation et de Suggestion d'Itinéraire ---

def inserer_site_dans_itineraire(id_site_a_inserer: int, df_itineraire_actuel: pd.DataFrame,
                                  df_tous_les_sites: pd.DataFrame, id_site_precedent_dans_itineraire: int) -> pd.DataFrame:
    """
    Insère un site spécifique dans un itinéraire existant après un site donné.
    Cette fonction met à jour les ordres des sites mais ne vérifie PAS la faisabilité.

    Args:
        id_site_a_inserer (int): ID du site à insérer.
        df_itineraire_actuel (pd.DataFrame): DataFrame de l'itinéraire actuel.
                                            Une COPIE DOIT être passée si l'original
                                            ne doit pas être modifié par cette fonction.
        df_tous_les_sites (pd.DataFrame): Informations complètes sur tous les sites.
                                          Doit contenir 'ID_Site', 'Nom', 'Horaires', 'Temps_PEC',
                                          'Maint_Prev', 'Maint_Corr', 'Ouv_Matin', 'Ferm_Matin', 'Ouv_Aprem', 'Ferm_Aprem'.
        id_site_precedent_dans_itineraire (int): ID du site qui précédera le nouveau site.
                                                 Utiliser 0 si le site doit être inséré au début.

    Returns:
        pd.DataFrame: Le nouvel itinéraire avec le site inséré, trié par 'Ordre'.
    """
    df_nouvel_itineraire = df_itineraire_actuel.copy()

    # Déterminer l'ordre où le nouveau site sera inséré
    ordre_insertion = 2
    if not df_nouvel_itineraire.empty:
        if id_site_precedent_dans_itineraire != 0:
            # Trouver l'ordre du site précédent
            if id_site_precedent_dans_itineraire in df_nouvel_itineraire['ID_Site'].values:
                ordre_precedent = df_nouvel_itineraire[df_nouvel_itineraire['ID_Site'] == id_site_precedent_dans_itineraire]['Ordre'].iloc[0]
                ordre_insertion = ordre_precedent + 1
            else:
                # Gérer le cas où le site précédent n'est pas trouvé (erreur ou itinéraire vide/mal formé)
                # Par défaut, insérer à la fin si le site précédent n'est pas dans l'itinéraire
                ordre_insertion = df_nouvel_itineraire['Ordre'].max() + 1
        else: # Insertion au tout début si id_site_precedent_dans_itineraire est 0
            ordre_insertion = 2
    else: # Si l'itinéraire est vide
        ordre_insertion = 2
    
    # Décaler les ordres des sites existants qui viendront après le nouveau site
    df_nouvel_itineraire.loc[df_nouvel_itineraire['Ordre'] >= ordre_insertion, 'Ordre'] += 1

    # Préparer le DataFrame pour le nouveau site
    # Assurez-vous que toutes les colonnes nécessaires sont présentes dans df_tous_les_sites
    # et que le site à insérer existe bien.
    if id_site_a_inserer not in df_tous_les_sites['ID_Site'].values:
        return df_nouvel_itineraire.sort_values(by='Ordre').reset_index(drop=True) # Retourne l'itinéraire non modifié
    
    site_info = df_tous_les_sites[df_tous_les_sites['ID_Site'] == id_site_a_inserer].iloc[0]
    nouveau_site_data = pd.DataFrame([{
        'Ordre': ordre_insertion,
        'Lieu': site_info['Nom'], # Assurez-vous que 'Nom' est dans site_info
        'ID_Site': site_info['ID_Site'],
        'Horaires': site_info['Horaires'], # Colonne pour affichage
        'Total Service': site_info['Temps_PEC'] + site_info.get('Maint_Prev', 0) + site_info.get('Maint_Corr', 0),
        'Heure_Debut': np.nan, # Placeholder, sera rempli par calculer_planning_apres_insertion
        'Heure_Fin': "",   # Placeholder
    }])

    # Concaténer le nouveau site et trier l'itinéraire par ordre
    df_nouvel_itineraire = pd.concat([df_nouvel_itineraire, nouveau_site_data], ignore_index=True)
    df_nouvel_itineraire = df_nouvel_itineraire.sort_values(by='Ordre').reset_index(drop=True)

    return df_nouvel_itineraire


def generer_suggestions_sites(df_itineraire_actuel: pd.DataFrame, df_tous_les_sites: pd.DataFrame,
                               df_durees_brut: pd.DataFrame, df_donnees_gps: pd.DataFrame,
                               chaine_horaires_technicien: str) -> pd.DataFrame:
    """
    Génère une liste de sites potentiels à suggérer pour l'ajout à l'itinéraire.
    Les suggestions sont basées sur la proximité géographique, les horaires d'ouverture/fermeture
    et la faisabilité de l'itinéraire résultant (vérifiée localement sans solveur complet).

    Args:
        df_itineraire_actuel (pd.DataFrame): DataFrame de l'itinéraire actuel.
                                            Contient 'ID_Site', 'Ordre', 'Total Service'
                                            et les colonnes d'horaires des sites.
        df_tous_les_sites (pd.DataFrame): Informations complètes sur tous les sites.
                                          Doit contenir 'ID_Site', 'Nom', 'Ouv_Matin', etc.
        df_durees_brut (pd.DataFrame): DataFrame original des durées (brut).
        df_donnees_gps (pd.DataFrame): Coordonnées GPS de tous les sites.
        chaine_horaires_technicien (str): Horaires du technicien sous forme de chaîne.

    Returns:
        pd.DataFrame: DataFrame des sites suggérés avec des colonnes comme 'ID_Site',
                      'temps_trajet_supplementaire', 'nom_site_precedent', 'id_site_precedent'.
                      Peut être vide si aucune suggestion n'est trouvée.
    """
    # Pré-traitement des données au début de cette fonction pour avoir les formats attendus
    matrice_durees = preparer_matrice_durees(df_durees_brut)
    horaires_technicien_minutes = parser_horaires_technicien(chaine_horaires_technicien)
    
    
    suggestions_trouvees = {}
    
    # Liste des IDs de sites déjà dans l'itinéraire (excluant les IDs virtuels comme 0)
    ids_deja_dans_itineraire = df_itineraire_actuel['ID_Site'].tolist()

    # Construction d'une liste d'IDs pour itérer sur les segments de l'itinéraire
    # Ajout de '0' au début et à la fin pour représenter les points de départ/arrivée virtuels
    df_itineraire_trie = df_itineraire_actuel.sort_values(by='Ordre').reset_index(drop=True)
    liste_ids_segments = [0] + df_itineraire_trie['ID_Site'].tolist() + [0] 



    # Itérer sur tous les segments possibles de l'itinéraire (points d'insertion)
    for i in range(len(liste_ids_segments) - 1):
        id_site_precedent_segment = liste_ids_segments[i]
        id_site_suivant_segment = liste_ids_segments[i+1]

        # --- Cas 1: Ajout au début de l'itinéraire (après le dépôt virtuel de départ) ---
        if id_site_precedent_segment == 0 and id_site_suivant_segment != 0:
            ids_candidats = trouver_sites_ouverture_tot_proches(
                id_site_suivant_segment, matrice_durees, df_tous_les_sites
            )
            for id_candidat in ids_candidats:

                if id_candidat not in ids_deja_dans_itineraire and df_tous_les_sites[df_tous_les_sites['ID_Site'] == id_candidat]['Ouv_Matin'].iloc[0] != 0:
                    # Tester l'insertion
                    df_itineraire_test = inserer_site_dans_itineraire(
                        id_candidat, df_itineraire_actuel.copy(), df_tous_les_sites, 0
                    )
                    # Vérifier la faisabilité avec la nouvelle fonction calculer_planning_apres_insertion
                    _, faisable, temps_perdu = calculer_planning_apres_insertion(df_itineraire_test, df_tous_les_sites, df_durees_brut, chaine_horaires_technicien)
                    
                    if faisable:
                        temps_sup = temps_perdu + calculer_temps_trajet_supplementaire(0, id_site_suivant_segment, id_candidat, matrice_durees)
                        if id_candidat not in suggestions_trouvees or suggestions_trouvees[id_candidat][0] > temps_sup:
                            suggestions_trouvees[id_candidat] = (temps_sup, "Départ", 0)
        # --- Cas 2: Ajout à la fin de l'itinéraire (avant le dépôt virtuel de fin) ---
        elif id_site_precedent_segment != 0 and id_site_suivant_segment == 0 and i == len(liste_ids_segments) - 2:
             ids_candidats = trouver_sites_fermeture_tard_proches(
                id_site_precedent_segment, matrice_durees, df_tous_les_sites
            )
             for id_candidat in ids_candidats:
                if id_candidat not in ids_deja_dans_itineraire:
                    # Tester l'insertion
                    df_itineraire_test = inserer_site_dans_itineraire(
                        id_candidat, df_itineraire_actuel.copy(), df_tous_les_sites, id_site_precedent_segment
                    )
                    # Vérifier la faisabilité avec la nouvelle fonction calculer_planning_apres_insertion
                    _, faisable,temps_perdu = calculer_planning_apres_insertion(df_itineraire_test, df_tous_les_sites, df_durees_brut, chaine_horaires_technicien)

                    if faisable:
                        temps_sup = temps_perdu + calculer_temps_trajet_supplementaire(id_site_precedent_segment, 0, id_candidat, matrice_durees)
                        if id_candidat not in suggestions_trouvees or suggestions_trouvees[id_candidat][0] > temps_sup:
                            nom_prec = df_tous_les_sites[df_tous_les_sites['ID_Site'] == id_site_precedent_segment]['Nom'].iloc[0]
                            suggestions_trouvees[id_candidat] = (temps_sup, nom_prec, id_site_precedent_segment)
        
        # --- Cas 3: Ajout entre deux sites réels de l'itinéraire ---
        elif id_site_precedent_segment != 0 and id_site_suivant_segment != 0:
            # Récupérer les coordonnées GPS des sites délimitant le segment
            lat_dep = df_donnees_gps[df_donnees_gps['ID_Site'] == id_site_precedent_segment]['latitude'].iloc[0]
            long_dep = df_donnees_gps[df_donnees_gps['ID_Site'] == id_site_precedent_segment]['longitude'].iloc[0]
            lat_arr = df_donnees_gps[df_donnees_gps['ID_Site'] == id_site_suivant_segment]['latitude'].iloc[0]
            long_arr = df_donnees_gps[df_donnees_gps['ID_Site'] == id_site_suivant_segment]['longitude'].iloc[0]

            # Calculer le point milieu et la longueur du segment pour définir une zone de recherche
            (centre_segment, longueur_segment) = trouver_milieu_segment_et_longueur(lat_dep, long_dep, lat_arr, long_arr)

            # Examiner les sites qui ne sont PAS encore inclus dans l'itinéraire
            df_sites_non_itineraire = df_tous_les_sites[~df_tous_les_sites['ID_Site'].isin(ids_deja_dans_itineraire)]

            for _, site_candidat_info in df_sites_non_itineraire.iterrows():
                id_candidat = site_candidat_info['ID_Site']
                lat_cand = df_donnees_gps[df_donnees_gps['ID_Site'] == id_candidat]['latitude'].iloc[0]
                long_cand = df_donnees_gps[df_donnees_gps['ID_Site'] == id_candidat]['longitude'].iloc[0]

                # Vérifier si le site candidat est géographiquement dans la "zone d'influence" du segment
                if est_point_dans_cercle(centre_segment, longueur_segment / 2, (long_cand, lat_cand)):
                    # Tester l'insertion
                    df_itineraire_test = inserer_site_dans_itineraire(
                        id_candidat, df_itineraire_actuel.copy(), df_tous_les_sites, id_site_precedent_segment
                    )
                    # Vérifier la faisabilité avec la nouvelle fonction calculer_planning_apres_insertion
                    _, faisable,temps_perdu = calculer_planning_apres_insertion(df_itineraire_test, df_tous_les_sites, df_durees_brut, chaine_horaires_technicien)

                    if faisable:
                        temps_sup = temps_perdu + calculer_temps_trajet_supplementaire(id_site_precedent_segment, id_site_suivant_segment, id_candidat, matrice_durees)
                        if id_candidat not in suggestions_trouvees or suggestions_trouvees[id_candidat][0] > temps_sup:
                            nom_prec = df_tous_les_sites[df_tous_les_sites['ID_Site'] == id_site_precedent_segment]['Nom'].iloc[0]
                            suggestions_trouvees[id_candidat] = (temps_sup, nom_prec, id_site_precedent_segment)

    # Convertir le dictionnaire des suggestions en DataFrame
    liste_suggestions_df = []
    for id_site, (temps_sup, nom_prec, id_prec) in suggestions_trouvees.items():
        liste_suggestions_df.append({
            "ID_Site": id_site,
            "temps_trajet_supplementaire": temps_sup,
            "nom_site_precedent": nom_prec,
            "id_site_precedent": id_prec
        })
    
    return pd.DataFrame(liste_suggestions_df)


def mettre_a_jour_suggestions_apres_insertion(id_site_nouvellement_insere: int, id_site_precedent_du_nouveau_site: int,
                                               df_suggestions_actuelles: pd.DataFrame, df_itineraire_mis_a_jour: pd.DataFrame,
                                               df_tous_les_sites: pd.DataFrame, df_durees_brut: pd.DataFrame,
                                               df_donnees_gps: pd.DataFrame, chaine_horaires_technicien: str) -> pd.DataFrame:
    """
    Met à jour la liste des suggestions après qu'un site a été inséré dans l'itinéraire.
    Pour simplifier et assurer la cohérence, cette fonction regénère l'intégralité
    des suggestions à partir du nouvel itinéraire.

    Args:
        id_site_nouvellement_insere (int): ID du site qui vient d'être ajouté (non utilisé directement ici, mais utile pour d'autres stratégies de mise à jour).
        id_site_precedent_du_nouveau_site (int): ID du site qui le précède dans l'itinéraire (non utilisé directement ici).
        df_suggestions_actuelles (pd.DataFrame): La liste des suggestions avant la mise à jour (non utilisée directement ici).
        df_itineraire_mis_a_jour (pd.DataFrame): L'itinéraire complet après l'insertion.
        df_tous_les_sites (pd.DataFrame): Informations complètes sur tous les sites.
        df_durees_brut (pd.DataFrame): DataFrame original des durées (brut).
        df_donnees_gps (pd.DataFrame): Coordonnées GPS de tous les sites.
        chaine_horaires_technicien (str): Horaires du technicien sous forme de chaîne.

    Returns:
        pd.DataFrame: La liste des suggestions entièrement regénérée et mise à jour.
    """
    # La stratégie la plus simple et la plus robuste est de regénérer toutes les suggestions.
    # Ceci assure que toutes les nouvelles contraintes et opportunités sont prises en compte.
    nouvelles_suggestions = generer_suggestions_sites(
        df_itineraire_mis_a_jour, df_tous_les_sites, df_durees_brut, df_donnees_gps, chaine_horaires_technicien
    )
    return nouvelles_suggestions


# --- 5. Fonction d'Optimisation de la Tournée Principale (pour remplissage automatique) ---

def optimiser_tournee_automatique_journee(df_itineraire_initial_optimise: pd.DataFrame, df_tous_les_sites: pd.DataFrame,
                                         df_durees_brut: pd.DataFrame, df_donnees_gps: pd.DataFrame,
                                         chaine_horaires_technicien: str,
                                         df_liste_sites_suggestion: pd.DataFrame,
                                         df_sites_courants: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise la tournée en ajoutant de manière automatique et itérative les sites
    non encore planifiés qui sont faisables et minimisent le temps de trajet supplémentaire,
    jusqu'à ce qu'aucun site supplémentaire ne puisse être ajouté sans rendre l'itinéraire
    infaisable ou sans dépasser la journée du technicien.

    Args:
        df_itineraire_initial_optimise (pd.DataFrame): L'itinéraire de base, déjà optimisé,
                                                     avec les colonnes 'ID_Site', 'Ordre',
                                                     'Total Service', 'Heure_Debut', 'Heure_Fin'
                                                     et les détails horaires des sites.
                                                     Si vide, le remplissage commencera à partir de zéro.
        df_tous_les_sites (pd.DataFrame): Informations complètes sur tous les sites,
                                          incluant 'ID_Site', 'Nom', 'Temps_PEC', 'Maint_Prev', 'Maint_Corr',
                                          'Ouv_Matin', 'Ferm_Matin', 'Ouv_Aprem', 'Ferm_Aprem'.
        df_durees_brut (pd.DataFrame): DataFrame original des durées de trajet.
        df_donnees_gps (pd.DataFrame): DataFrame des coordonnées GPS de tous les sites.
        chaine_horaires_technicien (str): Horaires du technicien sous forme de chaîne (ex: "08:00-17:30").

    Returns:
        pd.DataFrame: L'itinéraire final optimisé, avec tous les sites ajoutés et le planning calculé.
    """

   
    current_itineraire = df_itineraire_initial_optimise.copy()
    current_suggestion = df_liste_sites_suggestion.copy()

    current_sites_courants = df_sites_courants
    ajout_site = True 
    
    # Préparez la matrice des durées une seule fois
    matrice_durees = preparer_matrice_durees(df_durees_brut)

    

    while current_suggestion is not None and not current_suggestion.empty and ajout_site == True:
        ajout_site = False
        ids_in_suggestion = current_suggestion['ID_Site'].to_list()
        sites_suggerer = df_tous_les_sites[df_tous_les_sites['ID_Site'].isin(ids_in_suggestion)]
        ids_site_suggerer_avec_temps_PEC = sites_suggerer[sites_suggerer['Temps_PEC']>0]['ID_Site'].tolist()
        df_suggestions_triees = current_suggestion[current_suggestion['ID_Site'].isin(ids_site_suggerer_avec_temps_PEC)]
        df_suggestions_triees = df_suggestions_triees.sort_values(by='temps_trajet_supplementaire', ascending=True)
        
        if df_suggestions_triees is None or df_suggestions_triees.empty : 
            return current_itineraire, current_sites_courants

        meilleure_suggestion = df_suggestions_triees.iloc[0]
        
        df_itineraire_avec_ajout_manuel = inserer_site_dans_itineraire(
        id_site_a_inserer=meilleure_suggestion['ID_Site'],
        df_itineraire_actuel=current_itineraire.copy(), # Utilisez l'itinéraire optimisé actuel
        df_tous_les_sites=df_tous_les_sites,
        id_site_precedent_dans_itineraire=meilleure_suggestion['id_site_precedent']
        )
        # 2. Planifier et vérifier la faisabilité de ce nouvel itinéraire (localement)
        df_itineraire_planifie_manuel, faisable_manuel, _ = calculer_planning_apres_insertion(
            df_itineraire_avec_ajout_manuel,
            df_tous_les_sites, # df_tous_les_sites
            df_durees_brut, # df_durees_brut
            chaine_horaires_technicien # chaine_horaires_technicien
        )
        if faisable_manuel:
            ajout_site = True
            current_itineraire = df_itineraire_planifie_manuel.copy()
                                    
            ids_dans_nouvelle_tournee = current_itineraire['ID_Site'].unique()
            nouvelle_base_sites_df = df_tous_les_sites[df_tous_les_sites['ID_Site'].isin(ids_dans_nouvelle_tournee)].copy()
            valeurs_editables_actuelles = current_sites_courants[['ID_Site', 'Temps_PEC', 'Maint_Prev', 'Maint_Corr']].copy()

            temp_sites_courants = pd.merge(
                nouvelle_base_sites_df,
                valeurs_editables_actuelles,
                on='ID_Site',
                how='left',
                suffixes=('_defaut', '_edite') 
            )

            for col in ['Temps_PEC', 'Maint_Prev', 'Maint_Corr']:
                temp_sites_courants[col] = temp_sites_courants[f'{col}_edite'].fillna(temp_sites_courants[f'{col}_defaut'])
                temp_sites_courants = temp_sites_courants.drop(columns=[f'{col}_defaut', f'{col}_edite'])
                                

            current_sites_courants = pd.merge(
                temp_sites_courants,
                current_itineraire[['ID_Site', 'Heure_Debut', 'Heure_Fin', 'Ordre']],
                on='ID_Site',
                how='left'
            )

            current_sites_courants["Temps_Total_Service"] = \
            current_sites_courants["Temps_PEC"] + \
            current_sites_courants["Maint_Prev"] + \
            current_sites_courants["Maint_Corr"]
            
            def format_minutes_to_hhmm(minutes):
                if pd.isna(minutes): # Gérer les valeurs NaN si elles existent
                    return None
                heures = int(minutes // 60)
                minutes_restantes = int(minutes % 60)
                return f"{heures:02d}:{minutes_restantes:02d}"
                
            current_itineraire['Heure_Debut'] = current_itineraire['Heure_Debut'].apply(format_minutes_to_hhmm)
            current_itineraire['Heure_Fin'] = current_itineraire['Heure_Fin'].apply(format_minutes_to_hhmm) 



            current_suggestion = generer_suggestions_sites(current_itineraire, df_tous_les_sites,df_durees_brut,df_donnees_gps,chaine_horaires_technicien)

        
    return current_itineraire, current_sites_courants


        