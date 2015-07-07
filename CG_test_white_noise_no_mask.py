import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import CG_functions as CG
import time
import sys
import camber as cb


plt.ion()
nside = np.int(sys.argv[1])
lmax = np.int(sys.argv[2])
n_iter  = np.int(sys.argv[3])

params = cb.ini2dic("DX11_adi_params.ini")

print "nside = ", nside, "lmax = ",lmax,"n_iter = ",n_iter
### load data
### define cl, noise, and beam from commander/planck   

#beam from commander
#beam_file = "dx11_v2_commander_int_beam_005a_2048.fits"
#cl = np.load("test_cl.npy")*2.7255**2*1e12
#bl = hp.read_cl(beam_file)

#Planck 2015 camb Cl
cl = cb.generate_spectrum(params)[:,1]

#Gaussian beam
bl = CG.gaussian_beam(np.size(cl),5)



# White noise with amplitude given by the mean of the noise power spectrum (commander 2015 I believe)
sigma_l = 1.7504523623688016e-16*1e12
sigma = hp.synfast(sigma_l*np.ones(2500),nside).std()

# random map and noise generated
map_in = hp.synfast(cl*bl[:np.size(cl)]**2,nside)

noise = hp.synfast(sigma_l*np.ones(2500),nside)
map = map_in+noise


def test_spectra():
    plt.plot(cl[:lmax]*bl[:lmax]**2, label="input cl")
    plt.plot(sigma_l*np.ones(lmax), label="input nl")
    plt.plot(hp.anafast(map,lmax = 50), label="map Cl")
    plt.legend()
    hp.mollview(map,title = "input map")
    plt.show()


# filtering out the ell's above lmax (could have been done before)
map_in_filt = hp.alm2map(hp.map2alm(map,lmax=lmax),nside=nside)

# will be the N^{-1} from eq 25
inv_var = 1/sigma**2 *np.ones(hp.nside2npix(nside))

# first guess for the alm's
alm_guess = np.zeros((lmax+1)**2)

# define the first guess, the b from eq 25 since the code was design with this at first (might not be best idea ever)
Guess_dat = CG.data_class([alm_guess,cl,bl,sigma,inv_var,lmax,nside])
w_0 = CG.data_class([np.random.randn((lmax+1)**2),cl,bl,sigma,inv_var,lmax,nside])
w_1 = CG.data_class([0,cl,bl,sigma,inv_var,lmax,nside])
in_map = CG.data_class([0,cl,bl,sigma,inv_var,lmax,nside])

w1 = np.random.randn(hp.nside2npix(nside))
d = map

b_mf = CG.rs_data_matrix_func(in_map,d) 
b_w1 = CG.rs_w1_matrix_func(w_1,w1)
b_w0 = w_0.alm
b = b_mf + b_w1 + b_w0



###### run the CG algorithm (return map,cl and res)
out = CG.CG_algo(CG.A_matrix_func,b,Guess_dat,n_iter,0.00000000000000001)

out_mf = CG.CG_algo(CG.A_matrix_func,b_mf,Guess_dat,n_iter,0.00000000000000001)

out_w1 = CG.CG_algo(CG.A_matrix_func,b_w1,Guess_dat,n_iter,0.00000000000000001)

out_w0 = CG.CG_algo(CG.A_matrix_func,b_w0,Guess_dat,n_iter,0.00000000000000001)
