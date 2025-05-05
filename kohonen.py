# coding: utf8
#!/usr/bin/env python
# ------------------------------------------------------------------------
# Carte de Kohonen
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
# Implémentation de l'algorithme des cartes auto-organisatrices de Kohonen
# ------------------------------------------------------------------------
# Pour que les divisions soient toutes réelles (pas de division entière)
from __future__ import division

# Librairie de calcul matriciel
import numpy 
# Librairie d'affichage
import matplotlib.pyplot as plt


class Neuron:
  ''' Classe représentant un neurone '''
  
  def __init__(self, w, posx, posy):
    '''
    @summary: Création d'un neurone
    @param w: poids du neurone
    @type w: numpy array
    @param posx: position en x du neurone dans la carte
    @type posx: int
    @param posy: position en y du neurone dans la carte
    @type posy: int
    '''
    # Initialisation des poids
    self.weights = w.flatten()
    # Initialisation de la position
    self.posx = posx
    self.posy = posy
    # Initialisation de la sortie du neurone
    self.y = 0.
  
  def compute(self,x):
    '''
    @summary: Affecte à y la valeur de sortie du neurone (ici on choisit la distance entre son poids et l'entrée, i.e. une fonction d'aggrégation identité)
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # TODO
    self.y = numpy.linalg.norm(self.weights - x.flatten())**2 # Ligne modifiée

  def learn(self,eta,sigma,posxjetoile,posyjetoile,x):
    '''
    @summary: Modifie les poids selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param posxjetoile: position en x du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posxjetoile: int
    @param posyjetoile: position en y du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posyjetoile: int
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # TODO (attention à ne pas changer la partie à gauche du =)
    # Calcul de la distance au neurone gagnant dans la carte
    distance_carre = (self.posx - posxjetoile)**2 + (self.posy - posyjetoile)**2
    # Calcul du facteur de voisinage gaussien
    voisinage = numpy.exp(-distance_carre / (2 * sigma**2))
    # Mise à jour en place des poids du neurone
    self.weights[:] += eta * voisinage * (x.flatten() - self.weights)



class SOM:
  ''' Classe implémentant une carte de Kohonen. '''

  def __init__(self, inputsize, gridsize):
    '''
    @summary: Création du réseau
    @param inputsize: taille de l'entrée
    @type inputsize: tuple
    @param gridsize: taille de la carte
    @type gridsize: tuple
    '''
    # Initialisation de la taille de l'entrée
    self.inputsize = inputsize
    # Initialisation de la taille de la carte
    self.gridsize = gridsize
    # Création de la carte
    # Carte de neurones
    self.map = []    
    # Carte des poids
    self.weightsmap = []
    # Carte des activités
    self.activitymap = []
    for posx in range(gridsize[0]):
      mline = []
      wmline = []
      amline = []
      for posy in range(gridsize[1]):
        neuron = Neuron(numpy.random.random(self.inputsize),posx,posy)
        mline.append(neuron)
        wmline.append(neuron.weights)
        amline.append(neuron.y)
      self.map.append(mline)
      self.weightsmap.append(wmline)
      self.activitymap.append(amline)
    self.activitymap = numpy.array(self.activitymap)

  def compute(self,x):
    '''
    @summary: calcule de l'activité des neurones de la carte
    @param x: entrée de la carte (identique pour chaque neurone)
    @type x: numpy array
    '''
    # On demande à chaque neurone de calculer son activité et on met à jour la carte d'activité de la carte
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].compute(x)
        self.activitymap[posx][posy] = self.map[posx][posy].y

  def learn(self,eta,sigma,x):
    '''
    @summary: Modifie les poids de la carte selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param x: entrée de la carte
    @type x: numpy array
    '''
    # Calcul du neurone vainqueur
    jetoilex,jetoiley = numpy.unravel_index(numpy.argmin(self.activitymap),self.gridsize)
    # Mise à jour des poids de chaque neurone
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].learn(eta,sigma,jetoilex,jetoiley,x)     

  def scatter_plot(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée (utilisable dans le cas d'entrée à deux dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Modification des limites de l'affichage
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def scatter_plot_2(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée en 2 fois 2d (utilisable dans le cas d'entrée à quatre dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Affichage des 2 premières dimensions dans le plan
    plt.subplot(1,2,1)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Affichage des 2 dernières dimensions dans le plan
    plt.subplot(1,2,2)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,2].flatten(),w[:,:,3].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,2],w[i,:,3],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,2],w[:,i,3],'k',linewidth=1.)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def plot(self):
    '''
    @summary: Affichage des poids du réseau (matrice des poids)
    '''
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Création de la figure
    f,a = plt.subplots(w.shape[0],w.shape[1])    
    # Affichage des poids dans un sous graphique (suivant sa position de la SOM)
    for i in range(w.shape[0]):
      for j in range(w.shape[1]):
        plt.subplot(w.shape[0],w.shape[1],i*w.shape[1]+j+1)
        im = plt.imshow(w[i,j].reshape(self.inputsize),interpolation='nearest',vmin=numpy.min(w),vmax=numpy.max(w),cmap='binary')
        plt.xticks([])
        plt.yticks([])
    # Affichage de l'échelle
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace de la carte')
    # Affichage de la figure
    plt.show()

  def quantification(self,X):
    '''
    @summary: Calcul de l'erreur de quantification vectorielle moyenne du réseau sur le jeu de données
    @param X: le jeu de données
    @type X: numpy array
    '''
    # On récupère le nombre d'exemples
    nsamples = X.shape[0]
    # Somme des erreurs quadratiques
    s = 0
    # Pour tous les exemples du jeu de test
    for x in X:
      # On calcule la distance à chaque poids de neurone
      self.compute(x.flatten())
      # On rajoute la distance minimale au carré à la somme
      s += numpy.min(self.activitymap)**2
    # On renvoie l'erreur de quantification vectorielle moyenne
    return s/nsamples
  
# --------------------------Fonctions ajoutées---------------------------------
  
  def local_variation_simple(self):
    total = 0
    count = 0
    for x in range(self.gridsize[0]):
      for y in range(self.gridsize[1]):
        wj = self.map[x][y].weights
        voisins = []
        # for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-voisinage
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:  # 8-voisinage
          nx, ny = x + dx, y + dy
          if 0 <= nx < self.gridsize[0] and 0 <= ny < self.gridsize[1]:
            voisins.append(self.map[nx][ny].weights)
        if voisins:
          distances = [numpy.linalg.norm(wj - wi) for wi in voisins]
          total += sum(distances) / len(distances)
          count += 1
    return total / count if count > 0 else 0

  def local_variation_quadratique(self):
    total = 0
    count = 0
    for x in range(self.gridsize[0]):
      for y in range(self.gridsize[1]):
        wj = self.map[x][y].weights
        voisins = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-voisinage
        
          nx, ny = x + dx, y + dy
          if 0 <= nx < self.gridsize[0] and 0 <= ny < self.gridsize[1]:
            voisins.append(self.map[nx][ny].weights)
        if voisins:
          distances = [numpy.linalg.norm(wj - wi)**2 for wi in voisins]
          total += sum(distances) / len(distances)
          count += 1
    return total / count if count > 0 else 0

  # -----------------------------------------------------------------------------


# if __name__ == '__main__':
#     # Création d'un réseau avec une entrée (2,1) et une carte (10,10)
#     # TODO mettre à jour la taille des données d'entrée pour les données robotiques
#     CARTE = (10, 10)  # Taille de la carte
#     network = SOM((2, 1), CARTE)

# #   ETAS = [0.0, 0.5, 1, 2]
# #   SIGMAS = [0.1, 1.0, 2.0, 5.0]
# #   N_list = [100, 10000, 100000, 1000000]


# #   SIGMA_FIXE = 1.0
# #   N_FIXE = 10000
# #   CARTE_FIXE = (10, 10)
# #   ETA_FIXE = 0.01

#     # PARAMÈTRES DU RÉSEAU
#     ETA = 0.05  # Taux d'apprentissage
#     SIGMA = 1.5  # Largeur du voisinage
#     N = 500000  # Nombre de pas de temps d'apprentissage
#     VERBOSE = False  # TODO mettre à False pour que les simulations aillent plus vite
#     NAFFICHAGE = 1000  # Nombre de pas de temps avant rafraîchissement de l'affichage

#     # DONNÉES D'APPRENTISSAGE
#     nsamples = 1200

#     # Ensemble de données 1
#     samples = numpy.random.random((nsamples, 2, 1)) * 2 - 1

    
#     import datetime
#     now = datetime.datetime.now()
    

#     print("Test : ETA={ETA}, SIGMA={SIGMA}, N={N}, CARTE={CARTE}".format(ETA=ETA, SIGMA=SIGMA, N=N, CARTE=CARTE))
#     print("Date et heure actuelles : ", now.strftime("%Y-%m-%d %H:%M:%S"))

#     # TODO : Décommente un des ensembles ci-dessous si besoin

#     # Ensemble de données 2
#     # samples1 = -numpy.random.random((nsamples // 3, 2, 1))
#     # samples2 = numpy.random.random((nsamples // 3, 2, 1))
#     # samples2[:, 0, :] -= 1
#     # samples3 = numpy.random.random((nsamples // 3, 2, 1))
#     # samples3[:, 1, :] -= 1
#     # samples = numpy.concatenate((samples1, samples2, samples3))

#     # Ensemble de données 3
#     # samples1 = numpy.random.random((nsamples // 2, 2, 1))
#     # samples1[:, 0, :] -= 1
#     # samples2 = numpy.random.random((nsamples // 2, 2, 1))
#     # samples2[:, 1, :] -= 1
#     # samples = numpy.concatenate((samples1, samples2))

#     # Ensemble de données robotiques
#     # samples = numpy.random.random((nsamples, 4, 1))
#     # samples[:, 0:2, :] *= numpy.pi
#     # l1 = 0.7
#     # l2 = 0.3
#     # samples[:, 2, :] = l1 * numpy.cos(samples[:, 0, :]) + l2 * numpy.cos(samples[:, 0, :] + samples[:, 1, :])
#     # samples[:, 3, :] = l1 * numpy.sin(samples[:, 0, :]) + l2 * numpy.sin(samples[:, 0, :] + samples[:, 1, :])

#     # Affichage des données (pour les ensembles 1, 2 et 3)
#     plt.figure()
#     plt.scatter(samples[:, 0, 0], samples[:, 1, 0])
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#     plt.suptitle('Données apprentissage')
#     plt.show()

#     # Affichage des données (pour l'ensemble robotique)
#     # plt.figure()
#     # plt.subplot(1, 2, 1)
#     # plt.scatter(samples[:, 0, 0].flatten(), samples[:, 1, 0].flatten(), c='k')
#     # plt.subplot(1, 2, 2)
#     # plt.scatter(samples[:, 2, 0].flatten(), samples[:, 3, 0].flatten(), c='k')
#     # plt.suptitle('Données apprentissage')
#     # plt.show()

#     # SIMULATION
#     network.plot()

#     if VERBOSE:
#         plt.figure()
#         plt.ion()
#         plt.show()
#     else:
#         network.scatter_plot(False)

#     for i in range(N + 1):
#         index = numpy.random.randint(nsamples)
#         x = samples[index].flatten()
#         network.compute(x)
#         network.learn(ETA, SIGMA, x)

#         if VERBOSE and i % NAFFICHAGE == 0:
#             plt.clf()
#             # TODO : Remplacer par scatter_plot_2 pour les données robotiques
#             network.scatter_plot(True)
#             plt.pause(0.00001)
#             plt.draw()

#     if VERBOSE:
#         plt.ioff()
#     else:
#         network.scatter_plot(False)

#     network.plot()

#     print("Erreur de quantification vectorielle moyenne :", network.quantification(samples))
#     print("Mesure locale (simple) :", network.local_variation_simple())
#     print("Mesure locale (quadratique) :", network.local_variation_quadratique())



if __name__ == '__main__':
    # Création d'un réseau avec une entrée (2,1) et une carte (10,10)
    # TODO mettre à jour la taille des données d'entrée pour les données robotiques
    CARTE = (10, 10)  # Taille de la carte
    network = SOM((2, 1), CARTE)

#   ETAS = [0.0, 0.5, 1, 2]
#   SIGMAS = [0.1, 1.0, 2.0, 5.0]
#   N_list = [100, 10000, 100000, 1000000]


#   SIGMA_FIXE = 1.0
#   N_FIXE = 10000
#   CARTE_FIXE = (10, 10)
#   ETA_FIXE = 0.01

    # PARAMÈTRES DU RÉSEAU
    N = 1000000  # Nombre de pas de temps d'apprentissage
    ETA_INIT = 0.1
    SIGMA_INIT = 3.0 
    TAU_ETA = N / 2  # ou N/3 selon la rapidité souhaitée
    TAU_SIGMA = N / 2

   
    VERBOSE = False  # TODO mettre à False pour que les simulations aillent plus vite
    NAFFICHAGE = 1000  # Nombre de pas de temps avant rafraîchissement de l'affichage

    # DONNÉES D'APPRENTISSAGE
    nsamples = 1200

    # Ensemble de données 1
    samples = numpy.random.random((nsamples, 2, 1)) * 2 - 1

    
    import datetime
    now = datetime.datetime.now()
    

    # print("Test : ETA={ETA}, SIGMA={SIGMA}, N={N}, CARTE={CARTE}".format(ETA=ETA, SIGMA=SIGMA, N=N, CARTE=CARTE))
    print("Date et heure actuelles : ", now.strftime("%Y-%m-%d %H:%M:%S"))

    # TODO : Décommente un des ensembles ci-dessous si besoin

    # Ensemble de données 2
    # samples1 = -numpy.random.random((nsamples // 3, 2, 1))
    # samples2 = numpy.random.random((nsamples // 3, 2, 1))
    # samples2[:, 0, :] -= 1
    # samples3 = numpy.random.random((nsamples // 3, 2, 1))
    # samples3[:, 1, :] -= 1
    # samples = numpy.concatenate((samples1, samples2, samples3))

    # Ensemble de données 3
    # samples1 = numpy.random.random((nsamples // 2, 2, 1))
    # samples1[:, 0, :] -= 1
    # samples2 = numpy.random.random((nsamples // 2, 2, 1))
    # samples2[:, 1, :] -= 1
    # samples = numpy.concatenate((samples1, samples2))

    # Ensemble de données robotiques
    # samples = numpy.random.random((nsamples, 4, 1))
    # samples[:, 0:2, :] *= numpy.pi
    # l1 = 0.7
    # l2 = 0.3
    # samples[:, 2, :] = l1 * numpy.cos(samples[:, 0, :]) + l2 * numpy.cos(samples[:, 0, :] + samples[:, 1, :])
    # samples[:, 3, :] = l1 * numpy.sin(samples[:, 0, :]) + l2 * numpy.sin(samples[:, 0, :] + samples[:, 1, :])

    # Affichage des données (pour les ensembles 1, 2 et 3)
    plt.figure()
    plt.scatter(samples[:, 0, 0], samples[:, 1, 0])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.suptitle('Données apprentissage')
    plt.show()

    # Affichage des données (pour l'ensemble robotique)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.scatter(samples[:, 0, 0].flatten(), samples[:, 1, 0].flatten(), c='k')
    # plt.subplot(1, 2, 2)
    # plt.scatter(samples[:, 2, 0].flatten(), samples[:, 3, 0].flatten(), c='k')
    # plt.suptitle('Données apprentissage')
    # plt.show()

    # SIMULATION
    network.plot()

    if VERBOSE:
        plt.figure()
        plt.ion()
        plt.show()
    else:
        network.scatter_plot(False)

    for i in range(N + 1):
        eta = ETA_INIT * numpy.exp(-i / TAU_ETA)
        sigma = SIGMA_INIT * numpy.exp(-i / TAU_SIGMA)

        index = numpy.random.randint(nsamples)
        x = samples[index].flatten()
        network.compute(x)
        network.learn(eta, sigma, x)

        if VERBOSE and i % NAFFICHAGE == 0:
            plt.clf()
            # TODO : Remplacer par scatter_plot_2 pour les données robotiques
            network.scatter_plot(True)
            plt.pause(0.00001)
            plt.draw()

    if VERBOSE:
        plt.ioff()
    else:
        network.scatter_plot(False)

    network.plot()

    t = numpy.arange(N + 1)
    eta_vals = ETA_INIT * numpy.exp(-t / TAU_ETA)
    sigma_vals = SIGMA_INIT * numpy.exp(-t / TAU_SIGMA)

    plt.figure()
    plt.plot(t, eta_vals, label="η(t)")
    plt.plot(t, sigma_vals, label="σ(t)")
    plt.xlabel("Itérations")
    plt.ylabel("Valeur")
    plt.title("Évolution de η(t) et σ(t) au cours de l’apprentissage")
    plt.legend()
    plt.grid(True)
    plt.show()


    print("Erreur de quantification vectorielle moyenne :", network.quantification(samples))
    print("Mesure locale (simple) :", network.local_variation_simple())
    print("Mesure locale (quadratique) :", network.local_variation_quadratique())



# if __name__ == '__main__':

#   # -------------------------
#   # Paramètres à tester
#   # -------------------------
#   ETAS = [0.001, 0.01, 0.1, 1]
#   SIGMAS = [0.1, 1.0, 2.0, 5.0]
#   N_list = [100, 10000, 50000, 100000]
#   CARTES = [(20, 1), (10, 10), (20, 5)]

#   VERBOSE = False
#   NAFFICHAGE = 1000
#   PARAMETRE_TESTS = ["ETA", "SIGMA", "N", "CARTE"]

#   # -------------------------
#   # Jeu 1
#   # -------------------------
#   nsamples = 1200
#   samples = numpy.random.random((nsamples, 2, 1)) * 2 - 1

#   plt.figure()
#   plt.scatter(samples[:, 0, 0], samples[:, 1, 0])
#   plt.xlim(-1, 1)
#   plt.ylim(-1, 1)
#   plt.suptitle('Donnees apprentissage')
#   plt.savefig("donnees_apprentissage.png")
#   plt.show()

#   # -------------------------
#   # Boucle principale par paramètre
#   # -------------------------
#   for PARAMETRE_TEST in PARAMETRE_TESTS:
#       print(f"\n### TEST SUR {PARAMETRE_TEST} ###")

#       if PARAMETRE_TEST == "ETA":
#           SIGMA_FIXE = 1.0
#           N_FIXE = 10000
#           CARTE_FIXE = (10, 10)
#           for eta in ETAS:
#             dossier = creer_dossier("Jeu1", PARAMETRE_TEST, eta)
#             print(f"\n--- Test ETA={eta}, SIGMA={SIGMA_FIXE}, N={N_FIXE}, Carte={CARTE_FIXE} ---")
#             network = SOM((2, 1), CARTE_FIXE)

#             # Affichage des poids du réseau INITIAUX
#             network.plot()
#             nom_graphique_init = generer_nom_graphique("Jeu1", eta, SIGMA_FIXE, N_FIXE, CARTE_FIXE, "initial")
#             plt.savefig(os.path.join(dossier, nom_graphique_init))
#             plt.show()

      

#             # Initialisation de l'affichage interactif
#             if VERBOSE:
#                 plt.figure()
#                 plt.ion()
#                 plt.show()
#             else:
#                 network.scatter_plot(False)

#             # Boucle d'apprentissage
#             for i in range(N_FIXE + 1):
#                 index = numpy.random.randint(nsamples)
#                 x = samples[index].flatten()
#                 network.compute(x)
#                 network.learn(eta, SIGMA_FIXE, x)

#                 if VERBOSE and i % NAFFICHAGE == 0:
#                     plt.clf()
#                     network.scatter_plot(True)
#                     plt.pause(0.00001)
#                     plt.draw()

#             # Fin de l'affichage interactif
#             if VERBOSE:
#                 plt.ioff()
#             else:
#                 network.scatter_plot(False)

#             # Affichage des poids du réseau FINAUX
#             network.plot()
#             nom_graphique_final = generer_nom_graphique("Jeu1", eta, SIGMA_FIXE, N_FIXE, CARTE_FIXE, "final")
#             plt.savefig(os.path.join(dossier, nom_graphique_final))
#             plt.show()

#             # Mesures et affichage console
#             eq = network.quantification(samples)
#             vlocal_simple = network.local_variation_simple()
#             vlocal_quad = network.local_variation_quadratique()
#             print("erreur de quantification vectorielle moyenne ", eq)
#             print("Mesure locale (simple) : ", vlocal_simple)
#             print("Mesure locale (quadratique) : ", vlocal_quad)

#             # Sauvegarde CSV
#             resultats = [{
#                 'Carte': f"{CARTE_FIXE[0]}x{CARTE_FIXE[1]}",
#                 'ETA': eta,
#                 'SIGMA': SIGMA_FIXE,
#                 'N': N_FIXE,
#                 'Erreur_Quantification': round(eq, 4),
#                 'V_local_simple': round(vlocal_simple, 4),
#                 'V_local_quadratique': round(vlocal_quad, 4)
#             }]
#             nom_fichier = generer_nom_fichier("Jeu1", eta, SIGMA_FIXE, N_FIXE, CARTE_FIXE)
#             enregistrer_resultats_csv(os.path.join(dossier, nom_fichier), resultats)
