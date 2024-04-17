import numpy as np

B = 3 #tesla                                                                                                                                                                                                              
amu = 0.00116591
c = 3E8
beta = 0.77 # 0.82 to give gamma factor from note                                                                                                                                                                         
gamma = 1/np.sqrt(1-beta**2)

print("gamma factor:", gamma)

Ef = amu*B*c*beta* (gamma**2)
print(Ef , "V/m = ", Ef /1000, "kV/m = ", Ef / (1000*100), "kV/cm")

p1 = beta*B
p2 = Ef/c

print(p1, p2)

hbar = 1.054E-34
e = 1.6E-19 #C                                                                                                                                                                                                            
m = 1.883E-28 #kg                                                                                                                                                                                                         

#input value from sim                                                                                                                                                                                                     
we = 0.1896E6 #2*np.pi / (40E-6) #-s   


print("omega_e:", we, "rads/s, giving T = ", 2*np.pi / we, "s")

nu = (2*m*we) / (e*(p1+p2))
#nuErr = (2*m/e*(p1+p2))*weErr

dmu = we * hbar / (2*(c * (p1+p2)))
#dmuErr = weErr * hbar / (c * (p1+p2))
print("nu: ", nu,  " dmu:", dmu, "C.m", dmu*100, "C.cm", dmu*100/e, "e.cm")