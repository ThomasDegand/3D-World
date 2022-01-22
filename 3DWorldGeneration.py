##################
## Importations ##
##################

from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from statistics import mean
from os import mkdir

################
## Constantes ##
################

coef = 18
L = coef*30 #Nombre pair
m = L//2
U = np.zeros((L,L,L), dtype = np.bool)
S = L*L*L
print("La liste des points de l'Univers a été créée")


###############
## Fonctions ##
###############

@jit
def newObject():
    return np.zeros((L,L,L), dtype = np.bool)

@jit
def ps(u,v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

@jit
def N(u):
    return ps(u, u)**0.5

@jit
def Proj(u,e):
    prodscal = ps(u,e)
    return (e[0]*prodscal, e[1]*prodscal, e[2]*prodscal)

@jit
def Norm(u):
    if N(u) != 0:
        return (u[0]/N(u), u[1]/N(u), u[2]/N(u))
    else:
        return (0, 0, 0)

@jit
def Ortho(u,v,w):
    e1 = Norm(u)
    f21 = Proj(v,e1)
    f2 = (v[0]-f21[0], v[1]-f21[1], v[2]-f21[2])
    e2 = Norm(f2)
    f31 = Proj(w,e1)
    f32 = Proj(w,e2)
    f3 = (w[0]-f31[0]-f32[0], w[1]-f31[1]-f32[1], w[2]-f31[2]-f32[2])
    e3 = Norm(f3)
    print("Trace :",e1[0]+e2[1]+e3[2])
    return (e1,e2,e3)

@jit
def OrthoY(u):
    (a, b, c) = u
    e1 = Norm(u)
    
    if a!=0 or c!=0:
        v = (-a*b, a*a + c*c, -b*c)
        e2 = Norm(v)
        w = (c, 0, -a)
        e3 = Norm(w)
        
    else:
        e2 = (0, 0, -1)
        e3 = (-1, 0, 0)

    return (e1,e2,e3)

@jit
def Rot(u,v,w,theta):
    theta = np.deg2rad(theta)
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
    Q = np.array([[u[0],v[0],w[0]],[u[1],v[1],w[1]],[u[2],v[2],w[2]]])
    P = np.dot(Q, R)
    up, vp, wp = (P[0][0], P[1][0], P[2][0]), (P[0][1], P[1][1], P[2][1]), (P[0][2], P[1][2], P[2][2])
    return (up,vp,wp)

@jit
def Translate(O, t):
    Out = newObject()
    print(Out)
    for x0 in range(L):
        for y0 in range(L):
            for z0 in range(L):
                x, y, z = x0+t[0], y0+t[1], z0+t[2]
                critA0 = (x >= 0) and (x < L)
                critA1 = (y >= 0) and (y < L)
                critA2 = (z >= 0) and (z < L)
                critB = critA0 and critA1 and critA2
                critC = (O[x0][y0][z0] == 1)
                if critB and critC:
                    Out[x][y][z] = 1
    return Out
    
@jit
def cube(c):
    c = c % (m+1)
    Cub = newObject()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                critA = lambda u: ((u >= m-c) and (u <= m+(c-1)))
                if critA(x) and critA(y) and critA(z):
                    Cub[x][y][z] = 1
    print("L'objet a bien été ajouté")
    return Cub

@jit
def sphere(r, e):
    Sph = newObject()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                mid = m - 0.5
                d = ((x-mid)**2 + (y-mid)**2 + (z-mid)**2)**0.5
                if  d - r <= e:
                    Sph[x][y][z] = 1
    print("L'objet a bien été ajouté")
    return Sph

@jit
def cylindreX(h, r, e):
    Cyl = newObject()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                mid = m - 0.5
                critA = lambda u: ((u >= m-h) and (u <= m+(h-1)))
                critB = ((y-mid)**2 + (z-mid)**2)**0.5 - r <= e
                if critA(x) and critB:
                    Cyl[x][y][z] = 1
    print("L'objet a bien été ajouté")
    return Cyl  

@jit
def cylindreY(h, r, e):
    Cyl = newObject()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                mid = m - 0.5
                critA = lambda u: ((u >= m-h) and (u <= m+(h-1)))
                critB = ((z-mid)**2 + (x-mid)**2)**0.5 - r <= e
                if critA(y) and critB:
                    Cyl[x][y][z] = 1
    print("L'objet a bien été ajouté")
    return Cyl   

@jit
def cylindreZ(h, r, e):
    Cyl = newObject()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                mid = m - 0.5
                critA = lambda u: ((u >= m-h) and (u <= m+(h-1)))
                critB = ((x-mid)**2 + (y-mid)**2)**0.5 - r <= e
                if critA(z) and critB:
                    Cyl[x][y][z] = 1
    print("L'objet a bien été ajouté")
    return Cyl

@jit
def camera(A, B, l, h, theta):
    mini = L*(3**0.5)
    maxi = 0
    image = np.zeros((l,h), dtype=float)
    ml = l//2
    mh = h//2
    (xA, yA, zA) = A
    (xB, yB, zB) = B
    
    u = (xB-xA, yB-yA, zB-zA)
    
##    (a, b, c) = u1
##    u2 = (-2*a, 1*b, 3*c)
##    u3 = (2*a, 2*b, 3*c)
##    e1, e2, e3 = Ortho(u)

    e1, e2, e3 = OrthoY(u) 
    #e1, e2, e3 = Rot(e1,e2,e3, theta)
    
    
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if U[x][y][z]:
                    v0 = (x-xA, y-yA, z-zA)
                    v1, v2, v3 = Proj(v0,e1), ps(v0,e2), ps(v0,e3)
                    X = int(ml+v2)
                    Y = int(mh+v3)
                    if X < l and Y < h:
                        norme = N(v0)
                        if (norme >= image[X][Y] or image[X][Y] == 0.0):
                            image[X][Y] = N(v0)
                            if norme < mini:
                                mini = norme
                            elif norme > maxi:
                                maxi = norme

    Minimum.append(mini)
    Maximum.append(maxi)
    return image
    

################
## Programmes ##
################

U += cube(coef*8)
U += cylindreY(coef*12, coef*5, 10)
l = 2*L
h = 2*L

##V = np.zeros((L,L,L,3), dtype=np.int8)
##for x in range(L):
##    for y in range(L):
##        for z in range(L):
##            V[x][y][z] = np.array([x, y, z])*U[x][y][z]
##V = np.reshape(V, (L**3,3))
##V = np.unique(V, axis=0)
##print("Les points sont enregistrés")



##capture = camera(A, B, l, h, 45)
##plt.imshow(capture)
##plt.show()

rep = 360
Capt = [0]*rep
Duration = []
Maximum = []
Minimum = []
theta = 0
print("Theta :",theta)

for i in range(rep):
    start = time()
    
    t = 2*(i/360)*np.pi
    A = (m*(1-np.sin(t)), L, m*(1-np.cos(t)))
    B = (m, 0, m)
    Capt[i] = camera(A, B, l, h, theta)

    
    end = time()
    duree = int((rep-i)*(end-start))
    Duration.append(duree)
    duree = int(mean(Duration[-3:]))
    secondes = str(duree%60)
    minutes = str((duree//60)%60)
    heures = str((duree//60)//60)
    print("Capture de", i, "/", rep-1, " | Temps restant : ", heures, "Heures", minutes, "Minutes et ", secondes, "Secondes")

dossier = str(time())
mkdir(dossier)
for i in range(rep):
    plt.imsave(fname=dossier+"/Capture_"+str(i).zfill(3)+".jpg", arr=Capt[i], vmin=min(Minimum), vmax=max(Maximum), cmap=cm.magma)
