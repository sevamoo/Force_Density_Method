# Force Density Method

A Python implementation of the Force Density Method (FDM), introduced by Schek (1974). 
- FDM is an algebraic approach for finding the shape (geometry) of structures under equilibrium of forces. 
- The FDM methods requires a connectivity graph (a given topology with fixed and free nodes), external loadings on the nodes (Vectors in 3 dimensions) and force densities on the edges as the main inputs.
- The solver then simply solves a system of linear equations and provides the 3-dimensional coordinates of the nodes under equilibrium.

![](Images/sign.png)

**Implemented by:** [Vahid Moosavi](https://www.vahidmoosavi.me)


- Implementation is the same as described in Schek (1974). 
- 	Schek, H. J. (1974). The force density method for form finding and computation of general networks. Computer methods in applied mechanics and engineering, 3(1), 115-134.


- Works easily with mesh-based structures or graphs (in Networkx format)
- As shown in examples, can be mixed with any machine learning algorithms, optimization methods (either gradient based or evolutionary strategies.)

- Look at the examples [here](https://nbviewer.jupyter.org/github/sevamoo/Force_Density_Method/tree/master/) 


- Some sample results:
	- Meshed shell structures 
	![](Images/mesh.png)
	- Parametrizing the force densities based on network centrality measures
	![](Images/7.png)
	![](Images/3.png)
	![](Images/4.png) 

	- A case of roof where the fixed nodes are on a circle with fixed radius.
	![](Images/8x8.png)

	- Clustering of 10K generated geometries based on the distribution of forces, edge length and load-paths.
	![](Images/SOM.png) 


# To-Dos:
- Integrating the FDM solver with an auto-differentiation framework such as [JAX](https://github.com/google/jax) or Tensorflow
_ Integration with subdivision rules and parametric topologies in addition to parametric force densities
- Integrating the FDM solver into an agent-based framework a Reinforcement Learning (RL) set up, where the agent will explore.


