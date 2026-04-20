import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk


import optimisation_tournee
import suggestions_sites # Assurez-vous que ce fichier contient les nouvelles fonctions

# --- FONCTIONS DE CHARGEMENT DES DONNÉES ---

@st.cache_data
def charger_dates_valides():
    """
    Lit le fichier d'horaires et retourne une liste des dates uniques et valides
    pour lesquelles il existe des données.
    """
    try:
        horaires_file = "synthese_horaires_sites.csv"
        df_horaires = pd.read_csv(horaires_file, sep=';', encoding='utf-8')
        
        # Convertit la colonne en dates, les formats invalides deviennent NaT (Not a Time)
        df_horaires['Date_calendrier'] = pd.to_datetime(df_horaires['Date_calendrier'], format='%d/%m/%Y', errors='coerce')
        
        # Supprime les lignes où la date n'a pas pu être interprétée
        df_horaires.dropna(subset=['Date_calendrier'], inplace=True)
        
        # Récupère les dates uniques et les trie
        dates_uniques = sorted(df_horaires['Date_calendrier'].unique())
        
        if dates_uniques:
            return dates_uniques[0].strftime("%d %m %Y"), dates_uniques[-1].strftime("%d %m %Y") # Retourne (min_date, max_date)
        return None, None
    except FileNotFoundError:
        st.error("Fichier `synthese_horaires_sites.csv` introuvable. Impossible de déterminer les dates valides.")
        return None, None

def check_mot_de_passe() : 
    """Vérifie si l'utilisateur a rentré le bon mdp"""

    def mot_de_passe_entered():
        """Vérifie le mdp tapé par l'utilisateur """
        if st.session_state["password"] == st.secrets["mot_de_passe"] :
            st.session_state["password_correct"] = True
            del st.session_state["password"]

        else : 
            st.session_state["password_correct"] = False
        
    if "password_correct" not in st.session_state : 
        st.text_input("Veuillez entrer un mot de passe :", type="password", on_change=mot_de_passe_entered, key="password")
        return False
    elif not st.session_state["password_correct"] : 
        st.text_input("Veuillez entrer un mot de passe :", type="password", on_change=mot_de_passe_entered, key="password")
        st.error("mot de passe incorrect ")
        return False
    else :
        return True 

def charger_donnees(date_selectionnee):
    try:
        sites_file = "sites.csv"
        durations_file = "durations.csv"
        distances_file = "distance.csv"
        home_site_durations_file = "durations_sites_maison.csv"
        tournees_file = "tournees.csv"
        horaires_file = "synthese_horaires_sites.csv"

        df_sites_original = pd.read_csv(sites_file, sep=';', encoding="latin-1")
        
        df_durees_temp = pd.read_csv(durations_file, sep=';', encoding='utf-8')
        df_durees_temp = df_durees_temp[df_durees_temp['id']>0]
        df_durees_temp = df_durees_temp.drop('nom',axis=1)
        df_durees_temp = df_durees_temp.drop('cluster',axis=1)

        df_distances_temp = pd.read_csv(distances_file, sep=';', encoding='utf-8')
        df_home_site_durations_temp = pd.read_csv(home_site_durations_file, sep=';', encoding='latin-1')
        df_tournees = pd.read_csv(tournees_file, sep=';', encoding='latin-1')
        df_horaires_temp = pd.read_csv(horaires_file, sep=';', encoding='utf-8')

        df_data_gps = df_sites_original.copy()
        
        
        df_data_gps = df_data_gps[['latitude','longitude']]
        df_data_gps['ID_Site']= df_sites_original['idSite']

    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}. Veuillez vérifier que tous les CSV sont présents.")
        return pd.DataFrame(),pd.DataFrame()
    
    

    df_horaires = df_horaires_temp.copy()
    df_horaires['Date_calendrier'] = pd.to_datetime(df_horaires['Date_calendrier'], format='%d/%m/%Y', errors='coerce')
    horaires_du_jour = df_horaires[df_horaires['Date_calendrier'].dt.date == date_selectionnee].copy()

    #Fusion des données
    df_merged = pd.merge(df_sites_original, df_tournees, left_on='cluster', right_on='numTournée', how='left')
    
    horaires_du_jour = horaires_du_jour.drop(['NomSite','Typologie MTK'], axis=1)
    df_merged = pd.merge(df_merged, horaires_du_jour, on='idSite', how='left')

    #CRÉATION DU DATAFRAME FINAL 'df_sites'
    df_sites = pd.DataFrame()
    df_sites["ID_Site"] = df_merged["idSite"]
    df_sites["Nom"] = df_merged["NomSite"]
    df_sites["Groupement"] = df_merged["nom"]
    print(df_merged['Nb_Heures'])
    temps_pec_heures = pd.to_numeric(df_merged["Nb_Heures"], errors='coerce')
    
    print(temps_pec_heures)
    df_sites["Temps_PEC"] = (temps_pec_heures * 60).astype(int)
    df_sites["Maint_Prev"] = 0 
    df_sites["Maint_Corr"] = 0 

    
    # Définition des horaires par défaut en minutes
    default_ouv_matin = 480  # 08:00
    default_ferm_matin = 720  # 12:00
    default_ouv_aprem = 810  # 13:30
    default_ferm_aprem = 1020 # 17:00
    default_horaires_str = "08:00-12:00 | 13:30-17:30 (Défaut)"

    # Condition : identifier les lignes où aucune info d'horaire n'a été trouvée
    sans_horaires_definis = pd.isna(df_merged['Date_calendrier'])


    # Pour l'ouverture du matin
    ouv_matin_reels = df_merged['Plage_horaire_1'].apply(optimisation_tournee.transformer_horaire.parser_plage_horaire).apply(lambda x: x[0])
    df_sites['Ouv_Matin'] = np.where(sans_horaires_definis, default_ouv_matin, ouv_matin_reels)

    # Pour la fermeture du matin
    ferm_matin_reels = df_merged['Plage_horaire_1'].apply(optimisation_tournee.transformer_horaire.parser_plage_horaire).apply(lambda x: x[1])
    df_sites['Ferm_Matin'] = np.where(sans_horaires_definis, default_ferm_matin, ferm_matin_reels)

    # Pour l'ouverture de l'après-midi
    ouv_aprem_reels = df_merged['Plage_horaire_2'].apply(optimisation_tournee.transformer_horaire.parser_plage_horaire).apply(lambda x: x[0])
    df_sites['Ouv_Aprem'] = np.where(sans_horaires_definis, default_ouv_aprem, ouv_aprem_reels)
    
    # Pour la fermeture de l'après-midi
    ferm_aprem_reels = df_merged['Plage_horaire_2'].apply(optimisation_tournee.transformer_horaire.parser_plage_horaire).apply(lambda x: x[1])
    df_sites['Ferm_Aprem'] = np.where(sans_horaires_definis, default_ferm_aprem, ferm_aprem_reels)

    # --- 6. CRÉATION DE LA COLONNE 'Horaires' LISIBLE (MODIFIÉ) ---
    def formater_horaires_display(row):
        h1 = row['Plage_horaire_1']
        h2 = row['Plage_horaire_2']
        if 'FERME' in str(h1).upper(): return "Fermé"
        horaires_str = str(h1).strip() if pd.notna(h1) else ""
        if pd.notna(h2) and str(h2).strip() != '': horaires_str += f" | {h2.strip()}"
        return horaires_str

    horaires_reels_str = df_merged.apply(formater_horaires_display, axis=1)
    df_sites['Horaires'] = np.where(sans_horaires_definis, default_horaires_str, horaires_reels_str)

    df_sites["Dans_Tournee_Defaut"] = False 



    return df_sites,df_durees_temp, df_data_gps



def charger_data_gps(liste_id):
    df_sites = pd.read_csv("sites.csv", sep=';', encoding="latin-1")
    df_sites = df_sites[df_sites['idSite'].isin(liste_id)]
    variables = ['latitude','longitude']
    df_data_gps = df_sites[variables]

    
    return df_data_gps

def format_minutes_to_hhmm(minutes):
    if pd.isna(minutes): # Gérer les valeurs NaN si elles existent
        return None
    heures = int(minutes // 60)
    minutes_restantes = int(minutes % 60)
    return f"{heures:02d}:{minutes_restantes:02d}"



# --- INTERFACE STREAMLIT ---
if check_mot_de_passe():
    st.success('Accès autorisé')
    st.set_page_config(page_title="Gestion Tournées Techniciens", layout="wide")
    min_date, max_date = charger_dates_valides()


    # Initialisation du session_state
    if 'horaire_tech' not in st.session_state : 
        st.session_state.horaire_tech = "08:00-17:30"
    if 'etape' not in st.session_state:
        st.session_state.etape = 1
    if 'sites_courants' not in st.session_state: # Contient les sites sélectionnés/optimisés avec tous les détails
        st.session_state.sites_courants = pd.DataFrame()
    if 'resultat_tournee' not in st.session_state: # Le DataFrame de l'itinéraire OPTIMISÉ (avec ordres, heures de début/fin)
        st.session_state.resultat_tournee = None
    if 'groupement_choisi' not in st.session_state:
        st.session_state.groupement_choisi = ""
    if 'site' not in st.session_state: # Contient TOUS les sites chargés (df_tous_les_sites)
        st.session_state.site = pd.DataFrame()
    if 'duration' not in st.session_state: # Contient le DataFrame brut des durées (df_durees_brut)
        st.session_state.duration = pd.DataFrame()
    if 'data_gps' not in st.session_state : # Contient les data_gps de TOUS les sites (df_donnees_gps)
        st.session_state.data_gps = pd.DataFrame()
    if 'tech' not in st.session_state:
        st.session_state.tech = ""
    # AJOUTÉ : Pour stocker la liste de suggestions générée
    if 'suggestions_actuelles' not in st.session_state: 
        st.session_state.suggestions_actuelles = pd.DataFrame()


    # --- ÉTAPE 1 : CHOIX DE LA DATE ---
    if st.session_state.etape == 1:
        st.header("Choix de la journée")
        st.subheader(f"Choisir une date entre : {min_date} et {max_date}")
        date = st.date_input("Date d'intervention")

    
        if st.button("✅ Valider cette date"):
            st.session_state.site,st.session_state.duration,st.session_state.data_gps = charger_donnees(date)
            
            st.session_state.etape = 2
            st.rerun()
        


    # --- ÉTAPE 2 : PARAMÉTRAGE ET SÉLECTION DES SITES ---
    if st.session_state.etape == 2:

        df_tournees = pd.read_csv("tournees.csv", sep=';', encoding='latin-1')
        df_techniciens = pd.read_csv("technicien.csv", sep=';', encoding='latin-1')
        df_techniciens['prenom nom']=df_techniciens['prenom'] + ' ' + df_techniciens['nom']

        st.header("Choix de la Tournée")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.tech = st.selectbox("Technicien", df_techniciens['prenom nom'].tolist())
            num_tournee =  df_techniciens[df_techniciens['prenom nom'] == st.session_state.tech]['tourne_defaut'].iloc[0]
            
        with col2:
            groupement = st.selectbox("Groupement géographique",df_tournees['nom'].tolist())
            st.session_state.groupement_choisi = groupement
            st.text(f"Zone géographique du technicien : { df_tournees[df_tournees['numTournée']==num_tournee]['nom'].iloc[0]}")
            
        st.subheader(f"Ajustement des interventions : {groupement}")
        col1, col2,_,_ = st.columns(4)
        with col1:
            st.text(f"⚠️Tous les temps sont en minutes \nLe technicien travaille 7h30 dans sa journée = 450 minutes")
        with col2:
            st.text(f"Temps_PEC par défaut est le temps (en minute) prévu pour la prise en charge")
        
        col_tournee, col_map = st.columns([4, 1])

        with col_tournee : 
            sites_du_groupe = st.session_state.site[st.session_state.site["Groupement"] == groupement].copy()
            
            sites_du_groupe["À_Visiter"] = sites_du_groupe["Dans_Tournee_Defaut"]
        
            colonnes_visibles_editor = ["À_Visiter", "Nom", "Horaires", "Temps_PEC", "Maint_Prev", "Maint_Corr"]
            edited_df = st.data_editor(sites_du_groupe[colonnes_visibles_editor], hide_index=True, width='stretch')

            liste_nom = edited_df[edited_df["À_Visiter"] == True]['Nom'].to_list()
            liste_id = sites_du_groupe[sites_du_groupe["Nom"].isin(liste_nom)]['ID_Site'].to_list()

            col1,_,col2,_ = st.columns(4)
            with col1:
                if st.button("🔄 Changer la date"):
                    st.session_state.etape = 1
                    st.session_state.sites_courants = pd.DataFrame()
                    st.session_state.resultat_tournee = None
                    st.session_state.suggestions_actuelles = pd.DataFrame() # Réinitialiser les suggestions
                    st.rerun()
            
            with col2 : 
                if st.button("🚀 Calculer l'itinéraire"):
                    sites_coches = edited_df[edited_df["À_Visiter"] == True].copy()
                    sites_coches["Temps_Total_Service"] = sites_coches["Temps_PEC"] + sites_coches["Maint_Prev"] + sites_coches["Maint_Corr"]

                    noms_choisis = sites_coches["Nom"].tolist()
                    details_sites = st.session_state.site[st.session_state.site["Nom"].isin(noms_choisis)][['ID_Site',"Nom", "Ouv_Matin", "Ferm_Matin", "Ouv_Aprem", "Ferm_Aprem"]]
                    sites_finaux = pd.merge(sites_coches, details_sites, on="Nom")

                    st.session_state.sites_courants = sites_finaux
                    
                    
                    if st.session_state.sites_courants.empty:
                        st.warning("Veuillez sélectionner au moins un site à visiter.")

                    else:
                        st.session_state.resultat_tournee = optimisation_tournee.optimiser_tournee(
                            st.session_state.sites_courants,
                            st.session_state.duration,
                            st.session_state.horaire_tech)
                        
                        st.session_state.etape = 3
                        st.rerun()

        
        with col_map : 
            if liste_id != [] :
                data = charger_data_gps(liste_id)
                st.map(data)
                st.markdown(f"*Dezoomer pour voir les points*")
            else:
                st.info("Sélectionnez des sites pour les voir sur la carte.")


        
        
    # --- ÉTAPE 3 : ATELIER D'AJUSTEMENT ---
    elif st.session_state.etape == 3:

       
        st.header("Ajustement de la Tournée")
        
        # Le `tournee_courante` est maintenant `st.session_state.resultat_tournee`
        
        tournee_courante = st.session_state.resultat_tournee
        st.session_state.sites_courants["Temps_Total_Service"] = st.session_state.sites_courants["Temps_PEC"] + st.session_state.sites_courants["Maint_Prev"] + st.session_state.sites_courants["Maint_Corr"]

        col_tournee, col_suggestions = st.columns([3, 1])
    
        with col_tournee:
            colMatin,colAprem,_, _,_ = st.columns(5)
            with colMatin : 
                liste_matin = ["08:00","07:00","07:30","08:00","08:30","09:00","09:30","10:00","11:00","12:00","13:00","14:00"]
                liste_aprem = ["17:30","12:00","13:00","13:30","14:00","15:00","16:00","16:30","17:00","17:30","18:00"]
                matin = st.selectbox("Heure début de journée",liste_matin)
            
            with colAprem :   
                aprem = st.selectbox("Heure fin de journée",liste_aprem)


            st.markdown("*La pause du midi dure 1h30 entre 12h et 14h*.")

            st.session_state.horaire_tech = matin+"-"+aprem
                
            
            col1, _,col3 = st.columns([2,1,1])

            with col1 : 
                st.subheader("Planning calculé")
            
                
            with col3 : 
                with st.popover("Comprendre la logique d'optimisation") :
                    st.markdown(f"### Explications")
                    st.write("L'optimiseur calcule l'itinéraire le plus efficace en jonglant avec les temps de trajet, les temps d'intervention et les plages d'ouverture de chaque site et du technicien.")
                    st.write('**1. Construire la matinée la plus "rentable" :**' )
                    st.write("L'outil teste intelligemment différentes combinaisons de sites pour construire la matinée la plus productive possible. Il va chercher à :")
                    st.markdown("""
                    *   Prioriser les sites qui ne sont **ouverts que le matin**.
                    *   Maximiser le **nombre de sites visités** et le **temps de travail effectif** avant la pause déjeuner.
                    """)
                    st.write('**2. Organiser l\'après-midi :**')
                    st.write("Une fois la meilleure tournée du matin trouvée, il planifie le reste des visites en tenant compte de la fin de la dernière intervention, de la **pause déjeuner** (= 1h30), et des horaires des sites restants.")
                    st.write("")
                    st.write("Si plusieurs itinéraires sont possibles, l'outil privilégiera systématiquement celui qui permet de terminer la journée le **plus tôt**.")

            st.session_state.sites_courants["Heure_Debut"] = None
            st.session_state.sites_courants["Heure_Fin"] = None
            st.session_state.sites_courants["Ordre"] = None

            if (len(st.session_state.sites_courants) == 0) : 
                st.error("Aucun site dans la tournée")
            
            elif (len(st.session_state.sites_courants) == 1) : 
                tournee_courante = pd.DataFrame()
                tournee_courante['ID_Site'] = st.session_state.sites_courants['ID_Site']
                tournee_courante['Ordre'] = 1
                tournee_courante['Total Service'] = st.session_state.sites_courants['Temps_Total_Service']
                tournee_courante['Ouv_Matin'] = st.session_state.sites_courants['Ouv_Matin']
                tournee_courante['Ferm_Matin'] = st.session_state.sites_courants['Ferm_Matin']
                tournee_courante['Ouv_Aprem'] = st.session_state.sites_courants['Ouv_Aprem']
                tournee_courante['Ferm_Aprem'] = st.session_state.sites_courants['Ferm_Aprem']
                tournee_courante,_ ,_= suggestions_sites.calculer_planning_apres_insertion(tournee_courante,st.session_state.site,st.session_state.duration, st.session_state.horaire_tech)
                st.session_state.resultat_tournee = tournee_courante

                st.session_state.sites_courants = st.session_state.sites_courants.drop('Heure_Debut', axis=1)
                st.session_state.sites_courants = st.session_state.sites_courants.drop('Heure_Fin', axis=1)
                st.session_state.sites_courants = st.session_state.sites_courants.drop('Ordre', axis=1)
                colonnes_a_joindre = tournee_courante[['ID_Site', 'Heure_Debut', 'Heure_Fin','Ordre']]
                
                st.session_state.sites_courants = pd.merge(
                    st.session_state.sites_courants,
                    colonnes_a_joindre,
                    on='ID_Site',       # La colonne commune pour la correspondance
                    how='left'          # 'left' pour garder toutes les lignes de la table de gauche
                    
                )
                st.session_state.sites_courants.sort_values('Ordre', ascending=True, inplace=True)

                st.session_state.sites_courants['Heure_Debut'] = st.session_state.sites_courants['Heure_Debut'].apply(format_minutes_to_hhmm)
                st.session_state.sites_courants['Heure_Fin'] = st.session_state.sites_courants['Heure_Fin'].apply(format_minutes_to_hhmm)

                st.error("Un seul site")

            elif tournee_courante is not None:
                st.session_state.sites_courants = st.session_state.sites_courants.drop('Heure_Debut', axis=1)
                st.session_state.sites_courants = st.session_state.sites_courants.drop('Heure_Fin', axis=1)
                st.session_state.sites_courants = st.session_state.sites_courants.drop('Ordre', axis=1)
                colonnes_a_joindre = tournee_courante[['ID_Site', 'Heure_Debut', 'Heure_Fin','Ordre']]
                
                st.session_state.sites_courants = pd.merge(
                    st.session_state.sites_courants,
                    colonnes_a_joindre,
                    on='ID_Site',       # La colonne commune pour la correspondance
                    how='left'          # 'left' pour garder toutes les lignes de la table de gauche
                    
                )
                st.session_state.sites_courants.sort_values('Ordre', ascending=True, inplace=True)
                st.session_state.sites_courants['Heure_Debut'] = st.session_state.sites_courants['Heure_Debut'].apply(format_minutes_to_hhmm)
                st.session_state.sites_courants['Heure_Fin'] = st.session_state.sites_courants['Heure_Fin'].apply(format_minutes_to_hhmm)
            else:
                st.error("⚠️ Alerte : Le planning est surchargé ou les horaires ne permettent pas de tout caser.")

            colonnes_visibles = ["Nom", "Horaires", "Temps_PEC", "Maint_Prev", "Maint_Corr","Temps_Total_Service","Heure_Debut","Heure_Fin"]
            edited_df = st.data_editor(st.session_state.sites_courants[colonnes_visibles], hide_index=True, width='stretch')
            st.session_state.sites_courants["Temps_Total_Service"] = st.session_state.sites_courants["Temps_PEC"] + st.session_state.sites_courants["Maint_Prev"] + st.session_state.sites_courants["Maint_Corr"]
            

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⬅️ Modifier la sélection"):
                    st.session_state.etape = 2
                    st.session_state.resultat_tournee = None
                    st.session_state.suggestions_actuelles = pd.DataFrame()
                    st.rerun()
            with col_btn2:
                if st.button("🔄 Recalculer"):
                    edited_df["Temps_Total_Service"] = edited_df["Temps_PEC"] + edited_df["Maint_Prev"] + edited_df["Maint_Corr"]
                    noms_choisis = edited_df["Nom"].tolist()
                    details_sites = st.session_state.site[st.session_state.site["Nom"].isin(noms_choisis)][['ID_Site',"Nom", "Ouv_Matin", "Ferm_Matin", "Ouv_Aprem", "Ferm_Aprem"]]

                    sites_finaux = pd.merge(edited_df, details_sites, on="Nom")
                    st.session_state.sites_courants = sites_finaux

                    st.session_state.resultat_tournee = optimisation_tournee.optimiser_tournee(st.session_state.sites_courants,st.session_state.duration,st.session_state.horaire_tech)
                    
                    st.rerun()

                if st.button("✅ Valider ce planning"):
                    edited_df["Temps_Total_Service"] = edited_df["Temps_PEC"] + edited_df["Maint_Prev"] + edited_df["Maint_Corr"]
                    noms_choisis = edited_df["Nom"].tolist()
                    details_sites = st.session_state.site[st.session_state.site["Nom"].isin(noms_choisis)][['ID_Site',"Nom", "Ouv_Matin", "Ferm_Matin", "Ouv_Aprem", "Ferm_Aprem"]]

                    sites_finaux = pd.merge(edited_df, details_sites, on="Nom")
                    st.session_state.sites_courants = sites_finaux

                    st.session_state.resultat_tournee = tournee_courante
                    st.session_state.etape = 4
                    st.rerun()
                

        with col_suggestions:
            col1, col2 = st.columns(2)

            with col1 : 
                st.subheader("💡 Suggestions")
            with col2 : 
                with st.popover("Comprendre la logique de suggestions") : 
                    st.markdown(f"### Explications")
                    st.write('La suggestion de sites fonctionne en utilisant l\'itinéraire calculé à gauche :')
                    st.write('**1 : Identification des sites pertinents géographiquement :**' )
                    st.write("Entre deux points : Pour un trajet (par exemple, entre Grenoble et Lyon), l'outil dessine une \"zone de recherche\" entre ces deux villes et propose les sites qui s'y trouvent (comme Vienne ou Bourgoin-Jallieu).")
                    st.write("Début/Fin de journée : Il recherche aussi des sites qui ouvrent tôt ou ferment tard à proximité du premier et dernier sites.")
                    st.write('**2 : Calcul du temps de trajet supplémentaire**')
                    st.write('Ce temps représente uniquement le temps de conduite supplémentaire nécessaire pour ajouter le site à l\'itinéraire après le site indiqué, mais il ne prend pas en compte les horaires : c\'est l\'optimisation de la tournée qui se fait ensuite qui va calculer le réel coût de l\'ajout de ce site')

                        

            if st.session_state.resultat_tournee is not None and not st.session_state.resultat_tournee[st.session_state.resultat_tournee['ID_Site'] < 1000].empty or len(st.session_state.sites_courants) == 1:
                st.session_state.resultat_tournee= st.session_state.resultat_tournee[st.session_state.resultat_tournee['ID_Site'] < 1000]

                st.session_state.suggestions_actuelles = suggestions_sites.generer_suggestions_sites(
                    df_itineraire_actuel=st.session_state.resultat_tournee, # L'itinéraire optimisé actuel
                    df_tous_les_sites=st.session_state.site, # Tous les sites disponibles (df_sites)
                    df_durees_brut=st.session_state.duration, # DataFrame brut des durées
                    df_donnees_gps=st.session_state.data_gps, # Coordonnées GPS de tous les sites
                    chaine_horaires_technicien=st.session_state.horaire_tech # Horaires du technicien
                )


                if (st.session_state.suggestions_actuelles is None or st.session_state.suggestions_actuelles.empty) : 
                    df_suggestions_a_afficher = pd.DataFrame()
                    st.info('Aucun site à suggérer')

                else : 
                # Fusionner les détails des sites depuis df_tous_les_sites pour l'affichage des suggestions
                    df_suggestions_a_afficher = pd.merge(
                        st.session_state.suggestions_actuelles,
                        st.session_state.site[['ID_Site', 'Nom', 'Horaires', 'Temps_PEC', 'Groupement']],
                        on='ID_Site',
                        how='left'
                    )
                    df_suggestions_a_afficher.sort_values('temps_trajet_supplementaire', ascending=True, inplace=True)
            else:
                df_suggestions_a_afficher = pd.DataFrame()
            
            # MODIFIÉ : Appel à optimiser_tournee_automatique_journee
            if st.button("✨ Remplir la journée automatiquement", type="primary"):
                if st.session_state.resultat_tournee is None or st.session_state.resultat_tournee.empty:
                    st.warning("Veuillez d'abord calculer un itinéraire de base pour que l'auto-remplissage puisse démarrer.")
                else:
                    st.info("Lancement de l'optimisation automatique pour remplir la journée...")
                    
                    itineraire_apres_auto_remplissage, sites_courants_apres_remplissage = suggestions_sites.optimiser_tournee_automatique_journee(
                        df_itineraire_initial_optimise=st.session_state.resultat_tournee, # L'itinéraire actuel déjà optimisé
                        df_tous_les_sites=st.session_state.site,
                        df_durees_brut=st.session_state.duration,
                        df_donnees_gps=st.session_state.data_gps,
                        chaine_horaires_technicien=st.session_state.horaire_tech,
                        df_liste_sites_suggestion = st.session_state.suggestions_actuelles,
                        df_sites_courants= st.session_state.sites_courants
                    )
                    
                    st.session_state.resultat_tournee = itineraire_apres_auto_remplissage
                    st.session_state.sites_courants = sites_courants_apres_remplissage
                    st.rerun()


            # --- Affichage des suggestions ---
            
            if st.session_state.sites_courants['Heure_Debut'].isnull().sum() >0 or st.session_state.resultat_tournee.empty:
                st.info("Calculer un itinéraire valable pour voir les suggestions.")
            elif df_suggestions_a_afficher.empty:
                st.info("Aucune suggestion supplémentaire trouvée pour cet itinéraire.")
            else:
                for _, site_suggere in df_suggestions_a_afficher.iterrows():
                    with st.container(border=True):
                        # MODIFIÉ : Noms des colonnes après merge pour l'affichage
                        st.write(f"**{site_suggere['Nom']}**") 
                        st.caption(f"{site_suggere['Horaires']}")
                        st.caption(f"Durée PEC : {site_suggere['Temps_PEC']} min")
                        st.caption(f"Trajet supplémentaire : {site_suggere['temps_trajet_supplementaire']} min")
                        st.caption(f"Ajouté après : {site_suggere['nom_site_precedent']}")
                        
                        # MODIFIÉ : Logique pour le bouton "Ajouter à la journée"
                        if st.button(f"Ajouter à la journée", key=f"add_{site_suggere['ID_Site']}_{site_suggere['id_site_precedent']}"):
                            
                            # 1. Insérer le site dans l'itinéraire actuel (structure seulement)
                            df_itineraire_avec_ajout_manuel = suggestions_sites.inserer_site_dans_itineraire(
                                id_site_a_inserer=site_suggere['ID_Site'],
                                df_itineraire_actuel=st.session_state.resultat_tournee.copy(), # Utilisez l'itinéraire optimisé actuel
                                df_tous_les_sites=st.session_state.site,
                                id_site_precedent_dans_itineraire=site_suggere['id_site_precedent']
                            )
                            # 2. Planifier et vérifier la faisabilité de ce nouvel itinéraire (localement)
                            df_itineraire_planifie_manuel, faisable_manuel, _ = suggestions_sites.calculer_planning_apres_insertion(
                                df_itineraire_avec_ajout_manuel,
                                st.session_state.site, # df_tous_les_sites
                                st.session_state.duration, # df_durees_brut
                                st.session_state.horaire_tech # chaine_horaires_technicien
                            )


                            if faisable_manuel:

                                st.session_state.resultat_tournee = df_itineraire_planifie_manuel.copy()
                                
                                ids_dans_nouvelle_tournee = st.session_state.resultat_tournee['ID_Site'].unique()
                                nouvelle_base_sites_df = st.session_state.site[st.session_state.site['ID_Site'].isin(ids_dans_nouvelle_tournee)].copy()
                                valeurs_editables_actuelles = st.session_state.sites_courants[['ID_Site', 'Temps_PEC', 'Maint_Prev', 'Maint_Corr']].copy()

                                temp_sites_courants_avec_valeurs_editables = pd.merge(
                                    nouvelle_base_sites_df,
                                    valeurs_editables_actuelles,
                                    on='ID_Site',
                                    how='left',
                                    suffixes=('_defaut', '_edite') 
                                )

                                for col in ['Temps_PEC', 'Maint_Prev', 'Maint_Corr']:
                                    temp_sites_courants_avec_valeurs_editables[col] = temp_sites_courants_avec_valeurs_editables[f'{col}_edite'].fillna(temp_sites_courants_avec_valeurs_editables[f'{col}_defaut'])
                                    temp_sites_courants_avec_valeurs_editables = temp_sites_courants_avec_valeurs_editables.drop(columns=[f'{col}_defaut', f'{col}_edite'])
                                

                                st.session_state.sites_courants = pd.merge(
                                    temp_sites_courants_avec_valeurs_editables,
                                    st.session_state.resultat_tournee[['ID_Site', 'Heure_Debut', 'Heure_Fin', 'Ordre']],
                                    on='ID_Site',
                                    how='left'
                                )

                                st.session_state.sites_courants["Temps_Total_Service"] = \
                                    st.session_state.sites_courants["Temps_PEC"] + \
                                    st.session_state.sites_courants["Maint_Prev"] + \
                                    st.session_state.sites_courants["Maint_Corr"]
                                

                                

                                st.session_state.sites_courants['Heure_Debut'] = st.session_state.sites_courants['Heure_Debut'].apply(format_minutes_to_hhmm)
                                st.session_state.sites_courants['Heure_Fin'] = st.session_state.sites_courants['Heure_Fin'].apply(format_minutes_to_hhmm)

                                

                                st.session_state.resultat_tournee['Heure_Debut'] = st.session_state.resultat_tournee['Heure_Debut'].apply(format_minutes_to_hhmm)
                                st.session_state.resultat_tournee['Heure_Fin'] = st.session_state.resultat_tournee['Heure_Fin'].apply(format_minutes_to_hhmm)

                                
                            else:
                                st.error(f"Impossible d'ajouter le site {site_suggere['Nom']} : l'itinéraire deviendrait infaisable.")
                            
                            st.rerun()


    # --- ÉTAPE 4 : SAUVEGARDE ---
    elif st.session_state.etape == 4:
        st.header("Validation et Sauvegarde")
    
        if st.session_state.resultat_tournee is not None and not st.session_state.resultat_tournee.empty:
            st.success("La tournée est optimisée et prête à être transmise.")
            st.dataframe(st.session_state.sites_courants, width='stretch')
        
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("💾 Confirmer l'enregistrement"):
                    id_sites_a_modifier = st.session_state.sites_courants['ID_Site'].to_list()
                    
                    try:
                        df = pd.read_csv('sites.csv',delimiter=';')
                    except FileNotFoundError:
                        print("Le fichier 'sites.csv' n'a pas été trouvé.")
                        exit()
                    except Exception as e:
                        print(f"Une erreur s'est produite lors de la lecture du fichier CSV : {e}")
                        exit()
                    
                    print(df.columns)
                    # Vérifier si les colonnes 'idSite' et 'NB_Heures' existent
                    if 'idSite' not in df.columns or 'Nb_Heures' not in df.columns:
                        print("Le fichier CSV ne contient pas les colonnes 'idSite' et/ou 'Nb_Heures'.")
                        exit()
                    df['idSite'] = pd.to_numeric(df['idSite'], errors='coerce')

                    try:
                        df.to_csv('sites.csv', sep=';', index=False)  # index=False pour ne pas écrire l'index du DataFrame dans le fichier
                        print("Le fichier 'sites.csv' a été mis à jour avec succès.")
                    except Exception as e:
                        print(f"Une erreur s'est produite lors de l'écriture dans le fichier CSV : {e}")


                    
                    st.success("Temps PEC soustrait !")
            with col_f2:
                if st.button("🔄 Créer une autre tournée"):
                    st.session_state.etape = 1
                    st.session_state.sites_courants = pd.DataFrame()
                    st.session_state.resultat_tournee = None
                    st.session_state.suggestions_actuelles = pd.DataFrame()
                    st.rerun()

                
        else:
            st.warning("Aucune donnée à sauvegarder.")
            if st.button("Retour"):
                st.session_state.etape = 1
                st.rerun()
