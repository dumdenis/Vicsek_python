#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:17:45 2019

@author: dumontdenis
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import time

#Parameters
size_box=5,5 #size of the box
R=1 #Radius of interaction
N_part=10 #Number of particles
N_step=10 #Number of timesteps
V=0.05 #Velocity of each particles
eps=3.0 #Bruit sur l'alignement

def config_init(size_x,size_y,N):
    '''Generate a random configuration (x,y,theta) of N particules in a box of
    size given by size_x*size_y'''
    x=pd.DataFrame([np.random.random(N)*size_x])
    y=pd.DataFrame([np.random.random(N)*size_y])
    theta=pd.DataFrame([np.random.random(N)*2*np.pi])
    conf=pd.concat([x,y,theta],axis=0,ignore_index=True)
    conf=conf.T
    conf.columns=['x','y','theta']
    return conf

def distance(x,y):#remplacer par une fct d'une librairie
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def is_neighbour(x,y):
    '''if the particle y is in the range of interaction of the particle x return True, else return False'''
    if distance(x,y)<=R:
        return True
    else:
        return False

def alignment(conf,N,Radius):
    '''calculate the mean angle of the neighbour particule in a radius R for
    each particule in the config'''
    list_mean_angle=[]
    for i in range(N):
        part_a=conf.iloc[i]
        list_angle=[]
        for j in range(N):
            part_b=conf.iloc[j]
            #Deplacement de la paticule pour sonder les zones non consideree du a la periodicite
            #Cas ou la paticule est dans un coin de la boite
            if part_a['x']+Radius>size_box[0] and part_a['y']+Radius>size_box[1]:#Coin HD
                x0,y0=part_a['x'],part_a['y']#HD
                x1,y1=part_a['x'],part_a['y']-size_box[1] #BD
                x2,y2=part_a['x']-size_box[0],part_a['y']-size_box[1] #BG
                x3,y3=part_a['x']-size_box[0],part_a['y'] #HG
                for k in [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['x']+Radius>size_box[0] and part_a['y']-Radius<0:#Coin BD
                x0,y0=part_a['x'],part_a['y']#BD
                x1,y1=part_a['x']-size_box[0],part_a['y'] #BG
                x2,y2=part_a['x']-size_box[0],part_a['y']+size_box[1] #HG
                x3,y3=part_a['x'],part_a['y']+size_box[1] #HD
                for k in [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['x']-Radius<0 and part_a['y']+Radius>size_box[1]:#Coin HG
                x0,y0=part_a['x'],part_a['y']#HG
                x1,y1=part_a['x']+size_box[0],part_a['y'] #HD
                x2,y2=part_a['x']+size_box[0],part_a['y']-size_box[1] #BD
                x3,y3=part_a['x'],part_a['y']-size_box[1] #BD
                for k in [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['x']-Radius<0 and part_a['y']-Radius<0:#Coin BG
                x0,y0=part_a['x'],part_a['y']#BG
                x1,y1=part_a['x'],part_a['y']+size_box[1] #HG
                x2,y2=part_a['x']+size_box[0],part_a['y']+size_box[1] #HD
                x3,y3=part_a['x']+size_box[0],part_a['y'] #BD
                for k in [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            #Cas ou la paticule est sur le bord de la boite
            elif part_a['x']+Radius>size_box[0]:#Coté droit
                x0,y0=part_a['x'],part_a['y']#CD
                x1,y1=part_a['x']-size_box[0],part_a['y'] #CG
                for k in [[x0,y0],[x1,y1]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['x']-Radius<0:#Coté gauche
                x0,y0=part_a['x'],part_a['y']#CD
                x1,y1=part_a['x']+size_box[0],part_a['y'] #CD
                for k in [[x0,y0],[x1,y1]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['y']+Radius>size_box[1]:#Coté haut
                x0,y0=part_a['x'],part_a['y']#CD
                x1,y1=part_a['x'],part_a['y']-size_box[1] #CB
                for k in [[x0,y0],[x1,y1]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            elif part_a['y']-Radius<0:#Coté bas
                x0,y0=part_a['x'],part_a['y']#CB
                x1,y1=part_a['x'],part_a['y']+size_box[1] #CH
                for k in [[x0,y0],[x1,y1]]:
                    if is_neighbour([part_b['x'],part_b['y']],k):
                        list_angle.append(part_b['theta'])
            #cas ou la paticule est loin des bords de la boite
            else :
                if is_neighbour([part_b['x'],part_b['y']],[part_a['x'],part_a['y']]):
                    list_angle.append(part_b['theta'])
        mean_sin_theta=np.mean(np.sin(list_angle))
        mean_cos_theta=np.mean(np.cos(list_angle))
        if mean_sin_theta*mean_cos_theta >=0:#Evaluation de la vitesse moyenne en fct du cadran trigono
             if mean_sin_theta>0:
                 mean_angle=np.arctan(mean_sin_theta/mean_cos_theta)
             else:
                 mean_angle=np.arctan(mean_sin_theta/mean_cos_theta)+np.pi
        else:
             if mean_sin_theta<0:
                 mean_angle=np.arctan(mean_sin_theta/mean_cos_theta)+2*np.pi
             else:
                 mean_angle=np.arctan(mean_sin_theta/mean_cos_theta)+np.pi
        epsilon=rd.uniform(-eps/2,eps/2)
        list_mean_angle.append(np.mod(mean_angle+epsilon,2*np.pi))#modulo 2 pi pour rester ente 0 et 2pi avec le rajout du bruit
        #prend en compte la particule testée dans les voisins donc list_mean_angle jamais vide
    return pd.DataFrame(list_mean_angle)

def move(conf,N):
        conf['x']=np.mod(conf['x']+V*np.cos(conf['theta']),size_box[0])
        conf['y']=np.mod(conf['y']+V*np.sin(conf['theta']),size_box[1])

##################################### RUN ######################################

#INITIALISATION (generation random ou a partir d'un .csv pour un restart)
start_time = time.time()#Recuperation de l'heaure de demarrage
config=config_init(size_box[0],size_box[1],N_part)

#chemin="/Users/dumontdenis/Dropbox/Recherche/Simu Python/Vicsek/restart.csv"
#data=pd.read_csv(chemin,sep=',',header=0,index_col=None)
#config=pd.concat([data.iloc[:,1:4]],axis=1)

#Graph
fig=plt.figure(0,figsize=(8,8))
ax=plt.subplot(aspect='equal')
plt.axis((0,size_box[0],0,size_box[1]))
cmap_RB=cm.get_cmap('hsv')
plt.scatter(config['x'],config['y'],c=cmap_RB(config['theta']/(2*np.pi)))
#fig.colorbar(ax.pcolormesh(data,cmap=cmap_RB, rasterized=True, vmin=-4, vmax=4))
for i in range(len(config['x'])):
    circle=plt.Circle((config['x'][i],config['y'][i]),R,color='k',fill=False)
    ax.add_artist(circle)
plt.savefig('a'+str(0)+'.png')
plt.close('all')

#RUN
list_anv=[]#storgae of the anv
new_alignment=alignment(config,N_part,R)
config['theta']=new_alignment
print('start')
for t in range(N_step):
    move(config,N_part)
    new_alignment=alignment(config,N_part,R)
    config['theta']=new_alignment
    anv=1/N_part/V*np.linalg.norm((np.sum(V*np.sin(config['theta'])),np.sum(V*np.sin(config['theta'])))) #Averaged normalised velocity
#    print(anv)
    list_anv.append(anv)
    if np.mod(t+1,5)==0:
#    #if 0==0:
        print(t+1,"/"+str(N_step))
        plt.figure(t+1,figsize=(8,8))
        ax=plt.subplot(aspect='equal')
        plt.axis((0,size_box[0],0,size_box[1]))
        plt.scatter(config['x'],config['y'],c=cmap_RB(config['theta']/(2*np.pi)))
#        for i in range(len(config['x'])):
#            circle=plt.Circle((config['x'][i],config['y'][i]),R,color='k',fill=False)
#            ax.add_artist(circle)
        plt.savefig('a'+str(t+1)+'.png')
        plt.close('all')
config.to_csv('laststep.csv')#Sauvegarde la derniere occurence pour permettre un restart a partir de celle-ci
print("Temps d execution (h) : "+str((time.time()-start_time)/60/24))

plt.plot(range(N_step),list_anv)
#########TODO##############
#Optimisation dataframe et vicsek2(calcul voisin)