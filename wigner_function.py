import qutip as qt, numpy as np, matplotlib.pyplot as plt, matplotlib.transforms as trans

def plot_wigner_psi_phi(psi,alpha_max = 7.5):
    """
    Funzione per visualizzare la wigner function e i suoi integrali sugli assi
    """
    fig = plt.figure(figsize=(9,9))

    widths = [6,3]
    heights = [6,3]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                              height_ratios=heights)

    x = np.linspace(-alpha_max,alpha_max,200) #determino griglia su cui valuto la funzione 
    wig = qt.wigner(psi,x,x) #calcolo la wigner function dello stato su due assi
    psi_x = np.sum(wig,axis=0) #integrale della funzione lungo un asse è probability distribution lungo l'altra 
    psi_p = np.sum(wig,axis=1)


    ax = fig.add_subplot(spec[0,0])
    qt.plot_wigner(psi,x,x,fig=fig,ax=ax)
    ax = fig.add_subplot(spec[0,1])
    base = plt.gca().transData
    rot = trans.Affine2D().rotate_deg(90)
    ax.plot(x,-psi_p, transform =  rot+base)
    ax.set_xticks([])
    ax.set_ylim(-alpha_max,alpha_max)
    
    ax = fig.add_subplot(spec[1,0])
    ax.plot(x,psi_x)
    ax.set_yticks([]);
    ax.set_xlim(-alpha_max,alpha_max)
    plt.show()

N = 1 #scelgo la dimensione dell'Hilbert space
psi = qt.fock(N,1) #creo un vettore di dimensione N che rappresenta un Fock state 
plot_wigner_psi_phi(psi)

N = 30
psi = qt.coherent(N,3)
plot_wigner_psi_phi(psi,alpha_max=7)
