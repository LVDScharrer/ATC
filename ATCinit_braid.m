function ATCinit_braid()
    sim = ATCengine;
    % sim.angA = 0;
    % sim.angB = amp;
    % sim.VX = 0;
    % sim.VY = 0;
    % sim.Wi = 2;
    % sim.x0 = 0;
    % sim.y0 = 0;
    % sim.R = 12;
    % sim.a0 = -5;
    % sim.Trun = Truns(ver);
    % sim.Tstop = Tstops(ver);
    % sim.angA = 0;
    % sim.angB = -1;
    % sim.a2 = 0.08; % 0.5
    % sim.Trun = 800;
    % sim.Tstop = 100;
    
    % sim.phi0 = pi*initAng;
    
    sim.eta = 25;
    % sim.lambda = 0;
    % sim.g = 0;
    
    sim.T = 6500;
    sim.outPeriod = 100;
    sim.pattern = "angBraid";
    sim.directory_name = "ATCdata_" + sim.pattern + "_T=" + sim.T + "_outP=" + sim.outPeriod;
    
    [Qxx,Qxy] = ATCengine.setUnifQ(sim.k2);
    % [Qxx,Qxy] = ATCengine.setSingleDefect(sim.x-sim.x0,sim.y-sim.y0,-0.5,0,0,0);
    % [Qxx,Qxy] = ATCengine.setSingleDefect(sim.x-sim.x0,sim.y-sim.y0,0.5,0,0,0.5*pi);
    % for m = 1:100 % Relax without flow to find a stable starting config
    %     [Qxx,Qxy] = ATCengine.relaxQ(Qxx,Qxy,sim.k2,sim.B,sim.dt);
    % end
    sim.initQxx = Qxx;
    sim.initQxy = Qxy;
    
    sim.evolve();
    
    % sim.tweezerAnim();
    sim.braidAnim();
    % sim.directorFlowAnim();
    
    close
end
% end
