# Active Topological Control Engine: A MATLAB Nematodynamics Simulator

ATCengine.m defines a simulation object in matlab that lets you evolve a 2D active nematic liquid crystal with a prescribed spatiotemporally varying activity field in time, with control over the initial texture of the nematic and placement of topological defects, and of the various fundamental dimensionless constants of the system.

ATCinit_braid.m is a representative example of an initialization of the simulation, used to accomplish a complex procedure manipulating topological defects. It begins by inducing the pair-production of two pairs of oppositely "charged" topological defects in the nematic texture, and then uses our newly proposed symmetry-based control method, "active topological tweezers", to precisely control the defects' motion, braiding their worldines into a figure-eight knot, ending with all 4 defects annihilating.

November 8 2026 update: The original version of this code did not have dealiasing in the nonlinear derivative terms, so I've added a new version with the 3/2 rule implemented. It is otherwise exactly the same, and the braiding procedure can be run with the dealiased version by just changing the "sim" variable in ATCinit_braid.
