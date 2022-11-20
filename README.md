# NN_rProp
NN_Rprop è un'implementazione di una rete neurale multistrato dinamica.
Permette di simulare la propagazione in avanti e dietro di una rete e cambiare i valori della stessa, come il numero di strati interni o il numero di nodi interni per ogni livello.
Come set di sample è stato utilizzato il dataset mnist, quindi in particolare, questa rete è implementata per risolvere un problema di classificazione a 10 classi.
Come algoritmo di aggiornamento pesi è stata utilizzata la RProp.
Si può modificare dal main le funzioni di attivazioni (Identità, sigmoide) per ogni layer, nonchè la funzione di errore (sum_of_squares,cross_entropy,cross_entropy_soft_max)
