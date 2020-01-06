# -*- coding: utf-8 -*-


# Vahid Moosavi 2019 12 31 21:26
# sevamoo@gmail.com
# https://www.vahidmoosavi.me/
# https://github.com/sevamoo/

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
# import tensorflow as tf
# from sklearn.manifold import TSNE
# import sompylib.sompy as SOM
from time import time
import random
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
import os
import glob
# %matplotlib inline


class FDM(object):
    def __init__(self,name=''):
        """
		name and data, neigh== Bubble or Guassian
		"""
        self.name = name;
    
    def solve_fdm_Sheck(self, Data_struct,plot_topologies=False):
    # D = C^T . Q. C
    # D_f = C^T . Q. C_f
        G = Data_struct['G']
        q = Data_struct['q']
        p_x = Data_struct['p_x']
        p_y = Data_struct['p_y']
        p_z = Data_struct['p_z']
        free_nodes_mask = Data_struct['free_nodes_mask']
        fixed_nodes = Data_struct['fixed_nodes']
        free_nodes = Data_struct['free_nodes']
        free_edges = Data_struct['free_edges']
        x_f = Data_struct['x_f']
        y_f = Data_struct['y_f']
        z_f = Data_struct['z_f']
        n_free_edges =  len(free_edges)
        n_nodes = len(G.node)
        C_s = np.zeros((n_free_edges,n_nodes))
        for i, e in enumerate(free_edges):
            if e[0] in fixed_nodes:
                C_s[i,e[0]] = -1
                C_s[i,e[1]] = 1
            elif e[0] in fixed_nodes:
                C_s[i,e[0]] = 1
                C_s[i,e[1]] = -1
            else:
                C_s[i,e[0]] = 1
                C_s[i,e[1]] = -1
        C = C_s[:,free_nodes]
        C_f = C_s[:,fixed_nodes]
        Q = np.eye(n_free_edges)
        np.fill_diagonal(Q, q)
        D = C.T.dot(Q).dot(C)
        D_f = C.T.dot(Q).dot(C_f)
        # if plot_topologies == True:
        #     print ('Cs,C,C_f')
        #     print (C_s.shape,C.shape,C_f.shape)
        #     print ('D,D_f')
        #     print (D.shape,D_f.shape)
        #     plt.rcParams.update({'font.size': 7})
        #     fig = plt.figure(figsize=(10,10))
        #     ax = plt.subplot(2,2,1)
        #     colormap = plt.cm.binary
        #     plt.imshow(C,cmap = colormap)
        #     plt.title('C')
            
        #     ax = plt.subplot(2,2,2)
        #     plt.imshow(C_f,cmap = colormap)
        #     plt.title('C_f')
            
        #     ax = plt.subplot(2,2,3)
        #     plt.imshow(D,cmap = colormap)
        #     plt.title('D')
            
        #     ax = plt.subplot(2,2,4)
        #     plt.imshow(D_f,cmap = colormap)
        #     plt.title('D_f')
            
        
        x_found = np.linalg.inv(D).dot(p_x - D_f.dot(x_f))
        y_found = np.linalg.inv(D).dot(p_y - D_f.dot(y_f))
        z_found = np.linalg.inv(D).dot(p_z - D_f.dot(z_f))
        
        
        final_positions = np.ones((len(free_nodes)+len(fixed_nodes),3))
        final_positions[free_nodes,0] = x_found
        final_positions[free_nodes,1] = y_found
        final_positions[free_nodes,2] = z_found

        final_positions[fixed_nodes,0] = x_f
        final_positions[fixed_nodes,1] = y_f
        final_positions[fixed_nodes,2] = z_f
        
        Data_struct['final_positions'] = final_positions
        Data_struct['D_f'] = D_f
        Data_struct['D'] = D
        Data_struct['C_f'] = C_f
        Data_struct['C'] = C        
        return Data_struct

    def plot_form_static(self,Data_struct,axs=None,xy_only=False):

        facecolor = 'black'
        if axs==None:
            fig = plt.figure(figsize=(10,5),facecolor=facecolor)
            ax1  = plt.subplot(1,2,1)
            ax1.set_aspect('equal', 'box')
            ax1.set_facecolor(facecolor)
            if xy_only == False:
                ax2  = plt.subplot(1,2,2)
                ax2.set_facecolor(facecolor)
                ax2.set_aspect('equal', 'box')
        else:
            ax1  = axs[0]
            ax1.set_facecolor(facecolor)
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            ax1.set_aspect('equal', 'box')
            
            if xy_only == False:
                ax2  = axs[1]
                ax2.set_facecolor(facecolor)
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax2.set_aspect('equal', 'box')

        G = Data_struct['G']
        xy_poses = Data_struct['all_poses']
        xy_poses = {}
        coords = Data_struct['final_positions'].copy()
        for i in range(coords.shape[0]):
            xy_poses[i]=coords[i,[0,1]]
        G = Data_struct['G']
        BG =Data_struct['base_G']
        q = Data_struct['q']
        edge_colors = dict(BG.edges)
        for i,e in enumerate(G.edges):
            edge_colors[e] = q[i]
        for e in set(BG.edges).difference(G.edges):
            edge_colors[e] = 4.
        col = np.asarray(list(edge_colors.values())).copy()
    #         col = q.copy()
    #         col = weighted_measures[:,0]
        col[col>=0] = .85
        col[col<0] = .15
        nx.draw_networkx(BG,ax=ax1,
                     pos=xy_poses,
                     arrows=False,
    #                      edge_color='black',
                     edge_color=list(col),
                     edge_cmap = plt.cm.RdYlBu_r,
                     edge_vmin = 0,
                     edge_vmax = 1.,
                     width = 1,
                     with_labels=False,
                     font_color='white',
                     node_size=.1,
                     node_color = 'k',  
                     cmap=plt.cm.jet,
                     alpha=1,
                     )    

        if xy_only == False:
            xz_poses = {}
            coords = Data_struct['final_positions'].copy()
            for i in range(coords.shape[0]):
                xz_poses[i]=coords[i,[0,2]]
            nx.draw_networkx(BG,ax = ax2,
                         pos=xz_poses,
                         arrows=False,
                         edge_color=list(col),
        #                      edge_color='black',  

                         edge_cmap = plt.cm.RdYlBu_r,
                         edge_vmin = 0,
                         edge_vmax = 1.,
        #                      edge_cmap = plt.cm.bone_r,
        #                      edge_vmin = 0,
        #                      edge_vmax = 1.,
                         width = 1.,
                         with_labels=False,
                         font_color='white',
                         node_size=.1,
                         node_color = 'k',   
                         cmap=plt.cm.jet,
                         alpha=1,
                         )        

        plt.tight_layout()




    def plot_form_interactive(self, Data_struct,fixed_node_size=.05,color=None):
        G = Data_struct['G']
        q = Data_struct['q']
        p_x = Data_struct['p_x']
        p_y = Data_struct['p_y']
        p_z = Data_struct['p_z']
        free_nodes_mask = Data_struct['free_nodes_mask']
        fixed_nodes = Data_struct['fixed_nodes'].copy()
        free_nodes = Data_struct['free_nodes']
        free_edges = Data_struct['free_edges']
        fixed_edges = Data_struct['fixed_edges']
        final_positions = Data_struct['final_positions']
        free_edges = np.asarray(free_edges)
        fixed_edges = np.asarray(fixed_edges)
        q = Data_struct['q'].copy()
        fig = ipv.figure(width=400, height=400)
        ipv.style.use(['dark'])
        if color == None:
            col = q.copy()
            col[col>=0] = .85
            col[col<0] = .15
            col = plt.cm.RdYlBu_r(col)[:,:3]
            col_fixed = (0.8988850442137639, 0.3054978854286813, 0.20676662821991543)
        else:
            col = color
            col_fixed = (0.8988850442137639, 0.3054978854286813, 0.20676662821991543)
        ipv.plot_trisurf(final_positions[:,0], final_positions[:,1], final_positions[:,2], lines=free_edges, color = col)
        ipv.plot_trisurf(final_positions[:,0], final_positions[:,1], final_positions[:,2], lines=fixed_edges, color = col_fixed)
        ipv.scatter(final_positions[fixed_nodes,0], final_positions[fixed_nodes,1], final_positions[fixed_nodes,2], marker='sphere',size=fixed_node_size, color=col_fixed)
        eps = 0.5
        ipv.xlim(np.min(final_positions[:,0])-eps,np.max(final_positions[:,0])+eps)
        ipv.ylim(np.min(final_positions[:,1])-eps,np.max(final_positions[:,1])+eps)
        ipv.zlim(np.min(final_positions[:,2])-eps,np.max(final_positions[:,2])+eps)
        ipv.squarelim();
        ipv.show()

    def cal_loadpath(self, Data_struct):
        xyz = Data_struct['final_positions']
        G = Data_struct['G']
        edges = list(G.edges)
        origins = np.asarray([xyz[e[0]] for e in edges])
        endings = np.asarray([xyz[e[1]] for e in edges])
        (origins-endings).shape
        edgelenght = np.linalg.norm(origins-endings,axis=1)
        q = Data_struct['q']
        Data_struct['edgelenght'] = edgelenght
        edge_loadpath = np.multiply(edgelenght,np.abs(q))
        loadpath = edge_loadpath.sum()
        Data_struct['edgelenght'] = edgelenght
        Data_struct['loadpath'] = loadpath
        return Data_struct

    def graph_analytics(self, Data_struct,vis_analytics=True):
        free_edges = Data_struct['free_edges']
        G = Data_struct['G']
        #build the graph of each structure
        # There are many more
        # Clustering_coef = nx.average_clustering(G)
        # degrees = [G.degree[k] for k in G.nodes]
        # avg_degrees = nx.average_neighbor_degree(G)
        # average_neighbor_degrees = [avg_degrees[k] for k in G.nodes]
        ## The followings need to be weighted by distances maybe
        # betweenness_centrality = nx.betweenness_centrality(G)
        # betweenness_centrality = [betweenness_centrality[k] for k in G.nodes]
        # closeness_centrality = nx.closeness_centrality(G)
        # closeness_centrality = [closeness_centrality[k] for k in G.nodes]
        # degree_centrality = nx.degree_centrality(G)
        # degree_centrality = [degree_centrality[k] for k in G.nodes]
        edge_betweenness = nx.edge_betweenness(G)
        # edge_betweenness = [edge_betweenness[k] for k in edge_betweenness.keys()]
        # harmonic_centrality = nx.harmonic_centrality(G)
        # harmonic_centrality = [harmonic_centrality[k] for k in G.nodes]
        # pagerank = nx.pagerank(G)
        # pagerank = [pagerank[k] for k in G.nodes]

        # For dual graph
        LG = nx.line_graph(G)


        # Clustering_coef_l = nx.average_clustering(LG)
        degrees_l = nx.degree(LG)
        avg_degrees_l = nx.average_neighbor_degree(LG)
        betweenness_centrality_l = nx.betweenness_centrality(LG)
        closeness_centrality_l = nx.closeness_centrality(LG)
        degree_centrality_l = nx.degree_centrality(LG)
        harmonic_centrality_l = nx.harmonic_centrality(LG)
        pagerank_l = nx.pagerank(LG)


        edge_betweenness_sorted  = []


        degrees_l_sorted = []
        avg_degrees_l_sorted = []
        betweenness_centrality_l_sorted = []
        closeness_centrality_l_sorted = []
        degree_centrality_l_sorted = []
        harmonic_centrality_l_sorted = []
        pagerank_l_sorted = []
        for fe in free_edges:
            try:
                edge_betweenness_sorted.append(edge_betweenness[(fe[0],fe[1])])
            except:
                edge_betweenness_sorted.append(edge_betweenness[(fe[1],fe[0])])
            try:
                degrees_l_sorted.append(degrees_l[(fe[0],fe[1])])
            except:
                degrees_l_sorted.append(degrees_l[(fe[1],fe[0])])
            try:
                avg_degrees_l_sorted.append(avg_degrees_l[(fe[0],fe[1])])
            except:
                avg_degrees_l_sorted.append(avg_degrees_l[(fe[1],fe[0])])
            try:
                betweenness_centrality_l_sorted.append(betweenness_centrality_l[(fe[0],fe[1])])
            except:
                betweenness_centrality_l_sorted.append(betweenness_centrality_l[(fe[1],fe[0])])
            try:
                closeness_centrality_l_sorted.append(closeness_centrality_l[(fe[0],fe[1])])
            except:
                closeness_centrality_l_sorted.append(closeness_centrality_l[(fe[1],fe[0])])
            try:
                degree_centrality_l_sorted.append(degree_centrality_l[(fe[0],fe[1])])
            except:
                degree_centrality_l_sorted.append(degree_centrality_l[(fe[1],fe[0])])
            try:
                harmonic_centrality_l_sorted.append(harmonic_centrality_l[(fe[0],fe[1])])
            except:
                harmonic_centrality_l_sorted.append(harmonic_centrality_l[(fe[1],fe[0])])
            try:
                pagerank_l_sorted.append(pagerank_l[(fe[0],fe[1])])
            except:
                pagerank_l_sorted.append(pagerank_l[(fe[1],fe[0])])


        measures = {}
    #     measures['edge_betweenness'] = edge_betweenness_sorted
    #     measures['edge_degrees'] = degrees_l_sorted
        measures['edge_avg_degrees'] = avg_degrees_l_sorted
        measures['edge_betweenness_centrality'] = betweenness_centrality_l_sorted
        measures['edge_closeness_centrality'] = closeness_centrality_l_sorted
    #     measures['edge_degree_centrality'] = degree_centrality_l_sorted
        measures['edge_harmonic_centrality'] = harmonic_centrality_l_sorted
        measures['edge_pagerank'] = pagerank_l_sorted
        
        for k in measures.keys():
            mm = measures[k]
            if np.min(mm)==np.max(mm):
                mm = mm/np.max(mm)
            else:
                mm = (mm-np.min(mm))/(np.max(mm)-np.min(mm))
            measures[k] = mm
            
        Data_struct['graph_analytics'] = measures
        
        
        if vis_analytics == True:
            fig = plt.figure(figsize=(18,18))
            measures = Data_struct['graph_analytics']
            all_poses = Data_struct['all_poses']
            for i,m in enumerate(measures.keys()):
                plt.subplot(3,3,i+1)
                col = measures[m]
                nx.draw_networkx(G,
                             pos=all_poses,
                             arrows=False,
                             edge_color=col,
                             edge_cmap = plt.cm.bone_r,
                             edge_vmin = 0,
                             edge_vmax = 1.,
                             width = 4*col,
                             with_labels=False,
                             font_color='white',
                             node_size=1,
    #                          node_color=free_nodes_mask,
                             cmap=plt.cm.jet,
                             alpha=1)    
                plt.title('{}'.format(m.replace('_l_sorted','').replace('_sorted','')))
            
            plt.tight_layout()
        return Data_struct


    def prepare_graph(self, base_G,all_poses,fixed_nodes,free_nodes,free_nodes_mask,show_graph=True,node_size=20,node_label=True):
    
        G = base_G.copy()
        x_f = []
        y_f = []
        z_f = []
        for i,fn in enumerate(fixed_nodes):
            x_f.append(all_poses[fn][0])
            y_f.append(all_poses[fn][1])
        x_f = np.asarray(x_f)
        y_f = np.asarray(y_f)
        z_f = np.zeros(y_f.shape)  
        
        free_edges = []
        fixed_edges = []
        edgelist = [e for e in G.edges()]
        for i, e in enumerate(edgelist):
    #         e = np.sort(e)
            if (e[0] in fixed_nodes) and (e[1] in fixed_nodes):
                try:
                    G.remove_edge(e[0],e[1])
                    fixed_edges.append(list(e))
                except:
                    G.remove_edge(e[1],e[0])
                    fixed_edges.append(list(e))
                pass
            else:
                free_edges.append(list(e))
        free_edges = np.asarray(free_edges)

        n_edges = len(G.edges)
        n_free_nodes = len(free_nodes)
        n_free_edges =  len(free_edges)
        
        
        if show_graph == True:
            nx.draw_networkx(G,
                             pos=all_poses,
                             arrows=False,
                             edge_color='red',
                             edge_cmap = plt.cm.jet,
                             edge_vmin = 0,
                             edge_vmax = 1,
                             width = 1,
                             with_labels=node_label,
                             font_color='white',
                             node_size=node_size,
                             node_color=free_nodes_mask,
                             cmap=plt.cm.jet,
                             alpha=1)
        Data_struct = {}
        Data_struct['G'] = G
        Data_struct['base_G'] = base_G
        Data_struct['free_nodes_mask'] = free_nodes_mask
        Data_struct['fixed_nodes'] = fixed_nodes
        Data_struct['free_nodes'] = free_nodes
        Data_struct['free_edges'] = free_edges
        Data_struct['fixed_edges'] = fixed_edges
        Data_struct['x_f'] = x_f
        Data_struct['y_f'] = y_f
        Data_struct['z_f'] = z_f
        Data_struct['all_poses'] = all_poses

        return Data_struct



