#Accompanying script for "Tight Bounds for Tight Links: Ropelength of T (Q, Q) torus links"
#by Alex KLotz
#This script will compute the contour length of a toroidal helix with given major (R)
#and minor (r) radii and compare it to that of a straight helix with the same pitch 
#and a height (H) equivalent to the circumference of the torus. This ratio is slightly greater than 1
#and decreases as the major radius becomes much larger than the smaller.

#The second part calculates the integrated effect of many concentric helices, 
#with the number of helices in each layer, the radius of each layer, and the total height specified.


import numpy as np
pi=np.pi

n=1000 #points per helix
p=1 #repetitions
r=2 #minor radius of torus or radius of helix
R=4 #major radius of helix

#compute the ratio for a single helix

H=2*pi*R/p #height of helix
L_helix=np.sqrt(H**2+(2*pi*r)**2)*p #length of straight helix
theta=np.linspace(0,2*pi,n) #angle variable
#next 7 lines create a vector with the coordinates of the toroidal helix
x=(R+r*np.cos(p*theta))*np.cos(theta) 
y=(R+r*np.cos(p*theta))*np.sin(theta)
z=r*np.sin(p*theta)
torus=np.zeros([n,3])
torus[:,0]=x
torus[:,1]=y
torus[:,2]=z

#next 5 lines compute the total length

dr_vec=np.diff(torus,n=1,axis=0) #find tangent vector
dr2_vec=dr_vec**2 #square each component
dr2=np.sum(dr2_vec,axis=1) #sum to find squared magnitude
dr=np.sqrt(dr2) #square root to find magnitude
L_torus=np.sum(dr) #sum of each tangent magnitude gives total length


print("Straight helix: "+str(L_helix))
print("Toroidal helix:"+str(L_torus))
print("Torus/straight ratio:"+str(L_torus/L_helix))

#Compute the ratio for shells of concentric helices, with N_list helices per shell.
#This involve adding up the corrections for the outer shell and all the inner shells, weighted by the number of helices per shell.
#This will converge on the value in the paper (1.0058) as T becomes large

T=10  #total number of concentric shells (not including rod)

N_list=np.array(range(1,T+1))*4 #this is the 4-incremented number, can be substituted with other numbers
r_list=np.array(range(1,T+1))*2 #radii
R=0.913*N_list[-1] #using major radius from paper for the 4-incremented helix
#R=N_list[-1] #remove the 0.913 to recover the double-donut 4-increment correction
#Use T=10 and R=N_list[-1] to evaluate the claim "the correction falls below 1.005 when T ≥ 10"

H=2*pi*R/p
L_h=H  #including the central rod
L_t=H  #including the central circle
for i in range(0,T):
    r=r_list[i]
    x=(R+r*np.cos(p*theta))*np.cos(theta)
    y=(R+r*np.cos(p*theta))*np.sin(theta)
    z=r*np.sin(p*theta)
    torus=np.zeros([n,3])
    torus[:,0]=x
    torus[:,1]=y
    torus[:,2]=z
    L_t=L_t+N_list[i]*np.sum(np.sqrt(np.sum(np.diff(torus,axis=0)**2,axis=1)))
    L_h=L_h+N_list[i]*np.sqrt(H**2+(2*pi*r)**2)*p

print("Integrated torus/straight ratio: "+str(L_t/L_h))
    




