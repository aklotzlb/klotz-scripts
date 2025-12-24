"""
Ellipse intersection detector. Will determine how many times one ellipses passes through another. If they pass 1 time, they have Gauss linking number 1. 
If they pass 2 times, program returns 2 if ellipse A passes through ellipse B, and -2 if B passes through A.
With three ellipses, it can determine whether they form Borromean rings.
This script also includes a comparison with the Topoly package and a parallel Jones implementation, which require installation of those packages.
Scroll to line 240 for implementation. The main function is passcheck.
By Alex Klotz, Jonathan Strange, Ryan Blair. 2025.
"""
import numpy as np
from time import time
installedtopoly=1 #set this to 1 if you have topoly installed. "pip install topoly" will install it on most machines
    

def matgen(R,e,phi,vec,dR): 
    #this crates a 4x3 matrix where the first 3x3 is a transformation matrix of a circle and the fourth row is a displacement vector (dR)
    #inputs are radius of the circle to be transformed into an ellipse (R), the aspect ratio (e), its orientation angle in the XY plane (phi),
    #and its unit normal vector (vec). There is no exception handling for non-unit vectors, don't tempt it. 
    dx=dR[0]
    dy=dR[1]
    dz=dR[2]
    if R==0:
        R=1
    if e==0:
        temp=np.random.rand()
        e=(1+temp)/(1-temp)
    if phi==0:
        phi=np.random.rand()*np.pi
    
    M=np.zeros([4,3])    
    m1=np.matrix([[np.sqrt(e)*R, 0, 0],[0, R/np.sqrt(e), 0],[0, 0, 1]])
    m2=np.matrix([  [np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0],  [0,0,1] ]  )
    vec=vec[0]

    if vec[2]==1:
        m3=np.matrix([[1,0,0],[0,1,0],[0,0,1]]) #edge case of z-normal
    else:       
        zhat=np.array([0,0,1])
        v=np.cross(zhat,vec)
        s=np.linalg.norm(v)
        c=np.dot(zhat,np.transpose(vec))
        V=np.matrix([  [0,-v[2],v[1]],  [v[2], 0 ,-v[0]] ,[-v[1],v[0],0]])
        q=(1-c)/s**2
        t=np.multiply(q,np.matmul(V,V))
        m3=np.matrix([[1,0,0],[0,1,0],[0,0,1]])+V+t
    E=np.matmul(m3,np.matmul(m2,m1))
    M[0:3,0:3]=E
    M[3,0]=dx
    M[3,1]=dy
    M[3,2]=dz
    return M


def ellipsegen(M,n):
    #generates Cartesian coordinates of an ellipse with n points from matrix M
    #used for plotting, but not for calculation. Can be used for GLN calculation.
    step=2*np.pi/(n)
    th=np.arange(0,2*np.pi,step)
    x=np.cos(th)
    y=np.sin(th)
    z=0*th
    circle=np.zeros((n,3))
    circle[:,0]=np.transpose(x)
    circle[:,1]=np.transpose(y)
    circle[:,2]=np.transpose(z)
    E=M[0:3,0:3]
    ellipse=np.matmul(E,np.transpose(circle))
    ellipse[0,:]=ellipse[0,:]+M[3,0]
    ellipse[1,:]=ellipse[1,:]+M[3,1]
    ellipse[2,:]=ellipse[2,:]+M[3,2]
    return ellipse


def randunit():#returns random unit vector
        vec=np.random.normal(0, 1, size=(1, 3))
        vec=vec/np.linalg.norm(vec)
        return vec

def glncrude(E1,E2):#calculates the Gauss linking number from a discrete curve, using the crude tangent vector method. Only for comparison purposes, use the GLNtopo function for a more exact computation
    n=np.shape(E1)
    n=n[1]
    m=np.shape(E2)
    m=m[1]
    t1=np.diff(E1)
    t2=np.diff(E2)
    gln=0
    for i in range(0,n-1):
        for j in range(0,m-1):
            R=E1[:,i]-E2[:,j]
            term=np.dot(R,np.cross(t1[:,i],t2[:,j]))/np.linalg.norm(R)**3
            gln=gln+term
    gln=gln/(4*np.pi)
    return gln

def getmatrix(E): #this will turn a Cartesian ellipse into a transformation matrix
    com=np.mean(E,1)
    [evals,evecs]=np.linalg.eig(np.cov(E))
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    n=evecs[:,2]
    m3=matgen(1,1,np.pi,[n],[0,0,0])
    m3=m3[0:3,:]
    #m3i=np.linalg.inv(m3[0:3,:])
    Es=E
    Es[0,:]=E[0,:]-com[0]
    Es[1,:]=E[1,:]-com[1]
    Es[2,:]=E[2,:]-com[2]
    planar=np.matmul(m3,Es)
    planar=planar[0:2,:]
    [evals2,evecs2]=np.linalg.eig(np.cov(planar))
    th=0
    if evecs2[0,0]==0:
        th=np.pi/2
    else:
        th=np.arccos(evecs2[0,0])
    if evecs2[1,1]<0 and evecs2[0,1]<0:
        th=-th
    idx = evals2.argsort()[::-1]   
    evals2 = evals2[idx]
    evecs2 = evecs2[:,idx]          
    e=np.sqrt(evals[0]/evals[1])
    R=np.sqrt(evals[0]+evals[1])
    mat=matgen(R,e,th,[n],com)
    return mat
def passcheck(M1,M2,flipped):
    #this takes in two ellipse matrices and determines the number of passes. Each matrix should be 4x3, with the first three columns the transformation and the fourth the displacement.
    #the third parameter is a recursion tally, it should be zero in the function call
    m1=M1[0:3,:]
    m2=M2[0:3,:]
    dx1=M1[3,:]
    dx2=M2[3,:]
    m1i=np.linalg.inv(m1)
    com=np.matmul(m1i,dx1)  
    #terms for computing intersection of the two planes
    AA=(m1[0,2]*m2[0,0]+m1[1,2]*m2[1,0]+m1[2,2]*m2[2,0] ) #cos term
    BB=(m1[0,2]*m2[0,1]+m1[1,2]*m2[1,1]+m1[2,2]*m2[2,1] ) #sin term
    CC=(m1[0,2]*(dx2[0]-dx1[0])+m1[1,2]*(dx2[1]-dx1[1])+m1[2,2]*(dx2[2]-dx1[2])) #offset term
    minarg=np.arctan(BB/AA)    
    if minarg<0:
        i1temp=minarg+2*np.pi
        i2temp=minarg+np.pi
    else:
        i1temp=minarg
        i2temp=minarg+np.pi    
    v1temp=AA*np.cos(i1temp)+BB*np.sin(i1temp)+ CC
    v2temp=AA*np.cos(i2temp)+BB*np.sin(i2temp)+ CC
    #v1 and v2 are the maximum and minimum of one ellipse realtive to the plane of the other. If both are positive or negative, the two planes do not intersect and there are zero passes
    passes=0
    
    if v1temp*v2temp<0:
        #computing zeros of plane intersection equation to find angles along each ellipse that the intersection occurs
         sol1=np.arctan2( [((-AA*CC-np.sqrt(AA**2*BB**2+BB**4-BB**2*CC**2))/(AA**2+BB**2))], [((-CC+AA**2*CC/(AA**2+BB**2)+AA*np.sqrt(-BB**2*(-AA**2-BB**2+CC**2))/(AA**2+BB**2))/BB)])
         sol2=np.arctan2( [((-AA*CC+np.sqrt(AA**2*BB**2+BB**4-BB**2*CC**2))/(AA**2+BB**2))],[((-CC+AA**2*CC/(AA**2+BB**2)-AA*np.sqrt(-BB**2*(-AA**2-BB**2+CC**2))/(AA**2+BB**2))/BB)])
         sol1=sol1[0]
         sol2=sol2[0]
         z1=np.pi/2-sol1
         z2=np.pi/2-sol2
         #finding X and Y coordinates of plane intersections after a transformation has been applied to both ellipses that returns one of them to a unit circle (based on the inverse of its transformation matrix)
         X1=np.cos(z1)*(m1i[0,0]*m2[0,0]+m1i[0,1]*m2[1,0]+m1i[0,2]*m2[2,0])+np.sin(z1)*((m1i[0,0]*m2[0,1]+m1i[0,1]*m2[1,1]+m1i[0,2]*m2[2,1])  )+dx2[0]*m1i[0,0]+dx2[1]*m1i[0,1]+dx2[2]*m1i[0,2]-com[0]
         X2=np.cos(z2)*(m1i[0,0]*m2[0,0]+m1i[0,1]*m2[1,0]+m1i[0,2]*m2[2,0])+np.sin(z2)*((m1i[0,0]*m2[0,1]+m1i[0,1]*m2[1,1]+m1i[0,2]*m2[2,1])  )+dx2[0]*m1i[0,0]+dx2[1]*m1i[0,1]+dx2[2]*m1i[0,2]-com[0]
         Y1=np.cos(z1)*(m1i[1,0]*m2[0,0]+m1i[1,1]*m2[1,0]+m1i[1,2]*m2[2,0])+np.sin(z1)*((m1i[1,0]*m2[0,1]+m1i[1,1]*m2[1,1]+m1i[1,2]*m2[2,1])  )+dx2[0]*m1i[1,0]+dx2[1]*m1i[1,1]+dx2[2]*m1i[1,2]-com[1]        
         Y2=np.cos(z2)*(m1i[1,0]*m2[0,0]+m1i[1,1]*m2[1,0]+m1i[1,2]*m2[2,0])+np.sin(z2)*((m1i[1,0]*m2[0,1]+m1i[1,1]*m2[1,1]+m1i[1,2]*m2[2,1])  )+dx2[0]*m1i[1,0]+dx2[1]*m1i[1,1]+dx2[2]*m1i[1,2]-com[1]
        #radii of intersection points, if they are less than 1 then one ellipse passes through another at that point
         R1=X1**2+Y1**2
         R2=X2**2+Y2**2
         if R1<1:
             passes=passes+1 
         if R2<1:
             passes=passes+1
         if passes==0 and flipped==0: #if there are zero passes, recursivelly check if there B passes through A instead of A through B.
             passes=-passcheck(M2,M1,1)

    return passes


def triplecheck(M1,M2,M3): #checks if three ellipses meet at a common point
    dx1=M1[3,:]    
    dx2=M2[3,:]   
    dx3=M3[3,:]   
    M1=M1[0:3,:]
    M2=M2[0:3,:]
    M3=M3[0:3,:]
    n1=M1[:,2]
    n2=M2[:,2]
    n3=M3[:,2]
    A=np.array([[n1[0],n1[1],n1[2]],[n2[0],n2[1],n2[2]],[n3[0],n3[1],n3[2]]])
    iA=np.linalg.inv(A)
    b=np.array([n1[0]*dx1[0]+n1[1]*dx1[1]+n1[2]*dx1[2],n2[0]*dx2[0]+n2[1]*dx2[1]+n2[2]*dx2[2],n3[0]*dx3[0]+n3[1]*dx3[1]+n3[2]*dx3[2] ])
    IC=np.matmul(iA,np.transpose(b))
    i1=np.linalg.inv(M1)
    i2=np.linalg.inv(M2)
    i3=np.linalg.inv(M3)
    p1=np.matmul(i1,IC-np.transpose(dx1))
    p2=np.matmul(i2,IC-np.transpose(dx2))
    p3=np.matmul(i3,IC-np.transpose(dx3))
    if (p1[0]**2+p1[1]**2<1) and (p2[0]**2+p2[1]**2<1) and (p3[0]**2+p3[1]**2<1):
        return 1
    else:
        return 0  

#this uses Topoly to compute the Jones polynomial of three ellipses
def borrJones(M1,M2,M3):
    N=100
    c1=np.transpose(ellipsegen(M1,N))
    c2=np.transpose(ellipsegen(M2,N))
    c3=np.transpose(ellipsegen(M3,N))
    f = open("borro.xyz", "w")
    for i in range(0,N):
        f.write(str(i)+" "+str(c1[i,0])+" "+str(c1[i,1])+" "+str(c1[i,2])+"\n")
    f.write(str(0)+" "+str(c1[0,0])+" "+str(c1[0,1])+" "+str(c1[0,2])+"\n")
    f.write("X\n")
    for i in range(0,N):
        f.write(str(N+i)+" "+str(c2[i,0])+" "+str(c2[i,1])+" "+str(c2[i,2])+"\n")
    f.write(str(N)+" "+str(c2[0,0])+" "+str(c2[0,1])+" "+str(c2[0,2])+"\n")
    f.write("X\n")
    for i in range(0,N):
        f.write(str(2*N+i)+" "+str(c3[i,0])+" "+str(c3[i,1])+" "+str(c3[i,2])+"\n")
    f.write(str(2*N)+" "+str(c3[0,0])+" "+str(c3[0,1])+" "+str(c3[0,2])+"\n")
    f.close()
    borrs="borro.xyz"
    JP = jones(borrs,max_cross=30,closure=Closure.CLOSED)   
    return JP

#this uses Topoly to compute the Gauss linking number
def GLNtopo(M1,M2):
    N=100
    c1=np.transpose(ellipsegen(M1,N))
    c2=np.transpose(ellipsegen(M2,N))
    f = open("gln1.xyz", "w")
    g = open("gln2.xyz", "w")
    for i in range(0,N):
        f.write(str(i+1)+" "+str(c1[i,0])+" "+str(c1[i,1])+" "+str(c1[i,2])+"\n")
    for i in range(0,N):
        g.write(str(i+1)+" "+str(c2[i,0])+" "+str(c2[i,1])+" "+str(c2[i,2])+"\n")
    f.close()
    g.close()
    G=gln("gln1.xyz","gln2.xyz")
    return G
#three ellipses with aspect ratio 2, pi area, random orientation, centered within 0.2 of origin.
M1=(matgen(1,2,0,randunit(),0.2*np.random.rand(1,3)[0])) 
M2=(matgen(1,2,0,randunit(),0.2*np.random.rand(1,3)[0]))    
M3=(matgen(1,2,0,randunit(),0.2*np.random.rand(1,3)[0]))

#if you want guaranteed Borromean rings uncomment the following
# M1=(matgen(1,2,np.pi,np.array([[0,0,1]]),np.zeros([1,3])[0]))       
# M2=(matgen(1,2,np.pi/2,np.array([[0,1,0]]),np.zeros([1,3])[0]))    
# M3=(matgen(1,2,np.pi/2,np.array([[1,0,0]]),np.zeros([1,3])[0]))


p1=passcheck(M1,M2,0)
p2=passcheck(M2,M3,0)
p3=passcheck(M1,M3,0)
triplepierce=0
triplepoint=0
t1=time()
if (p1==2 and p2==2 and p3==-2) or (p1==-2 and p2==-2 and p3==2):
    triplepierce=1
    triplepoint=triplecheck(M1, M2, M3)
t2=time()

print("Ellipse intersection method:")
print("Ellipse 1 and 2 have "+str(p1)+" passages")
print("Ellipse 2 and 3 have "+str(p2)+" passages")
print("Ellipse 1 and 3 have "+str(p3)+" passages")

if triplepierce==1:
    if triplepoint==1:
        print("Triple point exists, ellipses are borromean")
    else:
        print("No triple point, ellipses are not Borromean")
elif np.abs(p1)+np.abs(p2)+np.abs(p3)==6:
    print("Ellipses are not pierced in the correct order for Borromean linking.")
else:
    print("Ellipses are not borromean")
t_out=(t2-t1)*1000
if t_out==0:
    print("Ellipse intersection checking time: <0.1 ms")
else:
    print("Ellipse intersection checking time:"+ str(t_out)+" ms")

print(" ")

if installedtopoly:
    from topoly import *
    G12=GLNtopo(M1, M2)
    G23=GLNtopo(M2, M3)
    G13=GLNtopo(M1, M3)
    t1=time()
    print("Topoly Jones Polynomial method:")

    print("Triple Jones polynomial determination: "+borrJones(M1, M2, M3))
    print("Jones checking time: "+str((time()-t1)*1000)+" ms")

    print("Gauss linking number between 1 and 2: " + str(G12))
    print("Gauss linking number between 2 and 3: " + str(G23))
    print("Gauss linking number between 1 and 3: " + str(G13))
kast=0
##The following section implements a parallel Jones polynomial computation written by Kasturi Barkataki and Eleni Panagiotou, found at hxxps://github.com/Parallel-Jones-Polynomial/ParallelJones . 
##It requires a modification of the file ParJones.py in order to be called from Python rather than bash. Doing so requires serial rather than parallel computation, which slows it down compared to its bash-call version. 
##If you are really interested in trying this and it is before 2052 you might just want to email Alex Klotz.
if kast>0:
    import ParJones
    E1=np.transpose(np.round(ellipsegen(M1, 10),4))
    E2=np.transpose(np.round(ellipsegen(M2, 10),4))
    E3=np.transpose(np.round(ellipsegen(M3, 10),4))
    t1=time()
    print(" ")
    print("Barkataki parallel Jones method:")
    print("Pairwise Jones determination of 12, 23, 13")
    inp1=str([E1.tolist(),E2.tolist()])
    print(ParJones.jones_execution(1,ParJones.str_to_np_array(inp1),True,False))
    inp2=str([E3.tolist(),E2.tolist()])
    print(ParJones.jones_execution(1,ParJones.str_to_np_array(inp2),True,False))
    inp3=str([E3.tolist(),E1.tolist()])
    print(ParJones.jones_execution(1,ParJones.str_to_np_array(inp3),True,False))
    inp123=str([E1.tolist(),E2.tolist(),E3.tolist()])
    print("Triple Jones determination")
    JP=ParJones.jones_execution(1,ParJones.str_to_np_array(inp123),True,False)
    print(JP)

    print("Barkataki method time: "+str(1000*(time()-t1))+" ms")


