classdef ATCengine
    properties
        T = 1800;
        Nx = 128;
        Ny = 128;
        dx = 0.5;
        B = 2; % Set B and g = 1?
        g = 0.5 ;
        lambda = 1.8;
        eta = 25;
        i = 0;
        dt = 0.1; 
        outPeriod = 50;
        useGPU = 1;
        directory_name = "ATCdata";
        initQxx;
        initQxy;
        % initQxx,initQxy = Q_constructor_2D(ones([256,256]),ones([256,256]),zeros([256,256])); % Perfectly ordered along x-axis
        %%%%%%%%%%%%%%%%%%%%%
        pattern = "0";
        %%%%%%%%%%%%%%%%%%%%%
        R;
        a0;
        a1;
        a1x;
        a1y;
        a2;
        a2x;
        a2y;
        x0;
        y0;
        VX;
        VY;
        Ws;
        Wi;
        Wo;
        angA;
        angB;
        freq;
        phi0;
        Trun;
        Tstop;
        wellWidth;
        polyDegree;
        plusDefectTrajX;
        plusDefectTrajY;
        minusDefectTrajX;
        minusDefectTrajY;
        umax;
        vortmax;
    end

    properties(Dependent)
        numSteps;
        gridLength;
        t;
        xx;
        x;
        y;
        k;
        kx;
        ky;
        k2;
        alpha;
    end

    methods
        %====================================================================================================================================================
        % CALL evolve() TO RUN SIM
        %====================================================================================================================================================
        function evolve(obj)
        mkdir(obj.directory_name)
        ATCengine.exportQ(obj.initQxx,obj.initQxy,0,obj.directory_name);
        if obj.useGPU == 1
            gQxx = gpuArray(obj.initQxx);
            gQxy = gpuArray(obj.initQxy);
            gkx = gpuArray(obj.kx);
            gky = gpuArray(obj.ky);
            gk2 = gpuArray(obj.k2);
        else
            gQxx = obj.initQxx;
            gQxy = obj.initQxy;
            gkx = obj.kx;
            gky = obj.ky;
            gk2 = obj.k2;
        end
        
        for ind = 1:obj.numSteps
            obj.i = ind;
            if obj.useGPU == 1
                galpha = gpuArray(obj.alpha);
            else
                galpha = obj.alpha;
            end
            [gQxx,gQxy] = ATCengine.flowEvolveQ(gQxx,gQxy,gkx,gky,gk2,galpha,obj.eta,obj.lambda,obj.B,obj.g,obj.dt);
            
            if mod(ind,obj.outPeriod) == 0
                outInd = ind/obj.outPeriod;
                outQxx = gather(gQxx);
                outQxy = gather(gQxy);
                ATCengine.exportQ(outQxx,outQxy,outInd,obj.directory_name);
                dispString = "T = " + num2str(obj.t) + "/" + num2str(obj.T);
                disp(dispString);
            end
        end
        simString = obj.directory_name + "/simData.mat";
        save(simString,"obj");
        end
        %====================================================================================================================================================

        %====================================================================================================================================================
        % DEPENDENT PROPERTY DEFINITIONS
        %====================================================================================================================================================
        function numSteps = get.numSteps(obj)
            numSteps = floor(obj.T / obj.dt);
        end

        function gridLength = get.gridLength(obj)
            gridLength = obj.Nx / obj.dx;
        end
        
        function t = get.t(obj)
            t = obj.i * obj.dt;
        end

        function xx = get.xx(obj)
            xx = ((0:obj.dx:(obj.Nx-0.5)) - obj.Nx/2);
        end

        function x = get.x(obj)
            [x,~] = meshgrid(obj.xx,obj.xx);
        end

        function y = get.y(obj)
            [~,y] = meshgrid(obj.xx,obj.xx);
        end

        function k = get.k(obj)
            k = ATCengine.wavenums(obj.xx);
        end

        function kx = get.kx(obj)
            [kx,~] = meshgrid(obj.k,obj.k);
        end

        function ky = get.ky(obj)
            [~,ky] = meshgrid(obj.k,obj.k);
        end

        function k2 = get.k2(obj)
            k2 = (obj.kx).^2 + (obj.ky).^2;
        end

        function alpha = get.alpha(obj)
        switch obj.pattern
            case "0" % This is the default pattern. Use it to insert whatever function of x,y,t you want.
                x = obj.x; % Default Domain: [-64:0.5:63.5]
                y = obj.y; % Default Domain: [-64:0.5:63.5]
                t = obj.t; % Domain: [0,obj.T]
                alpha = 0;  
            case "const"
                alpha = obj.a0.*ones(size(obj.x));
            case "strip"
                alpha = ATCengine.activity_strip(obj.x,obj.Ws,obj.Wi,obj.a0,obj.a1,obj.a2);
            case "polyWell"
                alpha = ATCengine.activity_polyWell(obj.x,obj.a0,obj.wellWidth,obj.polyDegree);
            case "quadTweezerLine"
                alpha = ATCengine.activity_quadTweezerLinear(obj.x,obj.t,obj.Nx,obj.x0,obj.y0,obj.VX,obj.VY,obj.a0,obj.a2x,obj.a2y,obj.R,obj.Wi);
            case "constTweezerLine"
                alpha = ATCengine.activity_constTweezerLinear(obj.x,obj.t,obj.Nx,obj.x0,obj.y0,obj.VX,obj.VY,obj.a0,obj.R,obj.Wi);
            case "parabL"
                alpha = ATCengine.activity_parabolicL(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.a1x,obj.a2,obj.Trun,obj.Tstop);
            case "hyperbolicArc"
                alpha = ATCengine.activity_hyperbolicArc(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.VY,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.a2,obj.Trun,obj.Tstop);
            case "surf"
                alpha = ATCengine.activity_surf(obj.x,obj.t,obj.Nx,obj.a0,obj.VX,obj.Ws,obj.Wi);
            case "oscillateInterface"
                alpha = ATCengine.activity_oscillateInterface(obj.x,obj.t,obj.a0,obj.freq,obj.Wi,obj.Wo);
            case "annulus"
                alpha = ATCengine.activity_annulus(obj.x,obj.y,obj.R,obj.Ws,obj.Wi,obj.a0);
            case "braid"
                alpha = ATCengine.activity_braid(obj.x,obj.y,obj.t);
            case "rampTweezer"
                alpha = ATCengine.activity_rampTweezer(obj.x,obj.y,obj.a0,obj.a1x,obj.a1y,obj.R,obj.Wi);
            case "annulRotate"
                alpha = ATCengine.activity_annulRotate(obj.x,obj.y,obj.t,obj.a0,obj.R,obj.Ws,obj.Wi,obj.phi0,obj.freq);
            case "ratchet"
                alpha = ATCengine.activity__ratchet(obj.x,obj.t);
            case "hourglass"
                alpha = ATCengine.activity_hourglassTweezer(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.a1x,obj.a2,obj.Trun,obj.Tstop);
            case "angHarmon"
                alpha = ATCengine.activity_angularHarmonic(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.angA,obj.angB,obj.Trun,obj.Tstop);
            case "L-arc"
                alpha = ATCengine.activity_angHarmLarc(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.Trun,obj.Tstop);
            case "V-arc"
                alpha = ATCengine.activity_angHarmVarc(obj.x,obj.y,obj.t,obj.Nx,obj.VX,obj.VY,obj.x0,obj.y0,obj.R,obj.Wi,obj.a0,obj.Trun,obj.Tstop);
            case "angLin"
                alpha = ATCengine.activity_angHarmLinear(obj.x,obj.y,obj.t,obj.Nx,obj.x0,obj.y0,obj.VX,obj.VY,obj.a0,obj.angA,obj.angB,obj.R,obj.Wi);
            case "angBraid"
                alpha = ATCengine.activity_angHarmBraid(obj.x,obj.y,obj.t);
        end
        end
        %====================================================================================================================================================
        
        %====================================================================================================================================================
        % VISUALIZATION
        %====================================================================================================================================================
        function file_name = tweezerAnim(obj)
        objStat = obj;
        objStat.VX = 0;
        objStat.VY = 0;
        objStat.x0 = 0;
        objStat.y0 = 0;
        
        % Recording Animation
        nameString = eraseBetween(obj.directory_name,1,"data_",'Boundaries','inclusive');
        file_name = char("TweezerAnim-" + nameString); 
        Nematic_write = VideoWriter(file_name);
        open(Nematic_write);
        % Counting Files
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;

        disp("Computing defect trajectories...");
        obj = obj.computeDefectTrajs();
        
        disp("Starting animation...");

        % framesToDisplay = 1:numFiles; % Default
        framesToDisplay = 1:5:numFiles; % Sped up a bit
        % framesToDisplay = [1,10,800,850,900,1700,1800]; % Snapshots for V-arc
        % framesToDisplay = [1,10,640,700,760,1400,1520]; % Snapshots for L-arc
        for j = framesToDisplay
            obj.i = (j-1)*obj.outPeriod;
            objStat.i = (j-1)*objStat.outPeriod;
            hold off
            fig1 = figure(1);
            % minXLim = -60;
            % maxXLim = 60;
            % minYLim = -30;
            % maxYLim = 30;
            minXLim = -30;
            maxXLim = 30;
            minYLim = -15;
            maxYLim = 15;
            re = 4;
            % VORTICITY + DIRECTOR FIELD %
            subplot(2,100,1:55);
            set(gcf, 'Position',[225, 28, 1650, 842]);
            obj.plotDirecVort(re);
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])

            % FLOW FIELD %
            subplot(2,100,101:155);
            set(gcf, 'Position',[225, 28, 1650, 842]);
            obj.plotFluidVel(re);
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])

            % CENTERED ACTIVITY PATTERN %
            sp1 = subplot(2,3,3);
            objStat.plotCMActivity()
            % find current position [x,y,width,height], set x-pos of second axes equal to first
            pos1 = get(sp1,'Position');
            pos1(1) = pos1(1) + 0.0175;
            set(sp1,'Position',pos1)

            % DEFECT TRAJECTORIES %
            sp2 = subplot(2,3,6);
            obj.plotDefectTrajs()

            % SAVE VIDEO FRAMES %
            drawnow
            F1 = getframe(fig1);
            writeVideo(Nematic_write, F1);
            % % Take high quality snapshots at different times
            % snapshotString = "Snap_fr=" + j + "_" + string(file_name);
            % print(fig1, '-depsc2', '-vector', char(snapshotString));
        end
        close(Nematic_write);
        close(fig1);
        file_name = string(file_name) + ".avi";
        end

        function file_name = braidAnim(obj)
        objStat = obj;
        objStat.VX = 0;
        objStat.VY = 0;
        objStat.x0 = 0;
        objStat.y0 = 0;
        
        % Recording Animation
        nameString = eraseBetween(obj.directory_name,1,"data_",'Boundaries','inclusive');
        file_name = char("Anim-" + nameString); 
        Nematic_write = VideoWriter(file_name);
        open(Nematic_write);
        % Counting Files
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;

        disp("Computing defect trajectories...");
        obj = obj.computeDefectTrajs();
        % s_plusDefectTrajX = load("plusDefectTrajX.mat","temp_plusDefectTrajX");
        % obj.plusDefectTrajX = s_plusDefectTrajX.temp_plusDefectTrajX;
        % s_plusDefectTrajY = load("plusDefectTrajY.mat","temp_plusDefectTrajY");
        % obj.plusDefectTrajY = s_plusDefectTrajY.temp_plusDefectTrajY;
        % s_minusDefectTrajX = load("minusDefectTrajX.mat","temp_minusDefectTrajX");
        % obj.minusDefectTrajX = s_minusDefectTrajX.temp_minusDefectTrajX;
        % s_minusDefectTrajY = load("minusDefectTrajY.mat","temp_minusDefectTrajY");
        % obj.minusDefectTrajY = s_minusDefectTrajY.temp_minusDefectTrajY;
        % s_umax = load("umax.mat","velMax");
        % obj.umax = s_umax.velMax;
        % s_vortmax = load("vortmax.mat","vorticityMax");
        % obj.vortmax = s_vortmax.vorticityMax;
        
        disp("Starting animation...");

        % framesToDisplay = 1:numFiles; % Default
        framesToDisplay = 1:10:numFiles; % Sped up a bit
        % framesToDisplay = [2,650]; % Snapshots
        % framesToDisplay = [1,20,30,50,100,150,200,250,300,350,400,450,500,550,600,650]; % Snapshots
        for j = framesToDisplay
            obj.i = (j-1)*obj.outPeriod;
            objStat.i = (j-1)*objStat.outPeriod;
            hold off
            fig1 = figure(1);
            % minXLim = -60;
            % maxXLim = 60;
            % minYLim = -30;
            % maxYLim = 30;
            minXLim = -64;
            maxXLim = 64;
            minYLim = -64;
            maxYLim = 64;
            re = 8;

            % DEFECT TRACKING %
            subplot(1,150,1:20);
            set(gcf, 'Position',[600, 200, 1800, 600]);
            obj.plotBraidTrajs()
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            zlim([0 7500])
            zticks([0,obj.T])

            % VORTICITY + DIRECTOR FIELD %
            subplot(1,150,32:82);
            set(gcf, 'Position',[600, 200, 1800, 600]);
            obj.plotDirecVort(re);
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])


            % FLOW FIELD %
            subplot(1,150,100:150);
            set(gcf, 'Position',[600, 200, 1800, 600]);
            obj.plotFluidVel(re);
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])

            % % CENTERED ACTIVITY PATTERN %
            % sp1 = subplot(2,3,3);
            % objStat.plotCMActivity()
            % % find current position [x,y,width,height], set x-pos of second axes equal to first
            % pos1 = get(sp1,'Position');
            % pos1(1) = pos1(1) + 0.0175;
            % set(sp1,'Position',pos1)
            % 
            % % DEFECT TRAJECTORIES %
            % sp2 = subplot(2,3,6);
            % obj.plotDefectTrajs()

            % SAVE VIDEO FRAMES %
            drawnow
            F1 = getframe(fig1);
            writeVideo(Nematic_write, F1);
            % Take high quality snapshots at different times
            snapshotString = "Snap_fr=" + j + "_" + string(file_name);
            print(fig1, '-depsc2', '-vector', char(snapshotString));
        end
        close(Nematic_write);
        close(fig1);
        file_name = string(file_name) + ".avi";
        end

        function file_name = directorFlowAnim(obj)
        % Recording Animation
        nameString = eraseBetween(obj.directory_name,1,"data_",'Boundaries','inclusive');
        file_name = char("DirFlowAnim-" + nameString);
        Nematic_write = VideoWriter(file_name);
        open(Nematic_write);
        % Counting Files
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;

        disp("Computing defect trajectories...");
        obj = obj.computeDefectTrajs();

        disp("Starting animation...");
        for j = 1:numFiles
            obj.i = (j-1)*obj.outPeriod;
            hold off
            fig1 = figure(1);
            minXLim = -64;
            maxXLim = 64;
            minYLim = -64;
            maxYLim = 64;

            % VORTICITY + DIRECTOR FIELD %
            subplot(1,2,1);
            set(gcf, 'Position',[225, 28, 1650, 842]);
            obj.plotDirecVort();
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])

            % FLOW FIELD %
            subplot(1,2,2);
            set(gcf, 'Position',[225, 28, 1650, 842]);
            obj.plotFluidVel();
            xlim([minXLim maxXLim]); 
            ylim([minYLim maxYLim]);
            pbaspect([maxXLim-minXLim maxYLim-minYLim 1])


            % SAVE VIDEO FRAMES %
            drawnow
            F1 = getframe(fig1);
            writeVideo(Nematic_write, F1);
            % % Take high quality snapshots at different times
            % snapshotString = "Snap_t=" + tspan(j) + "_" + string(file_name);
            % print(fig1, '-depsc2', '-vector', char(snapshotString));
        end
        close(Nematic_write);
        close(fig1);
        file_name = string(file_name) + ".avi";
        end

        function obj = computeDefectTrajs(obj)
        borderWidth = 24 ;
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;
        velMax = 0;
        vorticityMax = 0;
        nDefects = 1; % Number of each type of defect (total/2)
        if obj.pattern == "angBraid"
            nDefects = 2;
            borderWidth = 1;
        end
        temp_plusDefectTrajX = zeros(nDefects,numFiles);
        temp_plusDefectTrajY = zeros(nDefects,numFiles);
        temp_minusDefectTrajX = zeros(nDefects,numFiles);
        temp_minusDefectTrajY = zeros(nDefects,numFiles);
        for j = 1:numFiles   
            [Qxx,Qxy] = ATCengine.importQ(j-1,obj.directory_name);
            [ux,uy] = ATCengine.vel_field(Qxx,Qxy,obj.kx,obj.ky,obj.k2,obj.alpha,obj.eta,obj.lambda,obj.B,obj.g);
            u = (ux.^2 + uy.^2).^0.5;
            vort = ATCengine.vorticity(ux,uy,obj.kx,obj.ky);
            velMax = max([velMax,max(max(u))]);
            vorticityMax = max([vorticityMax,max(max(abs(vort)))]);
            theta=0.5*atan2(Qxy,Qxx);
            % Finding defects
            [temploc12,temploc_12] = ATCengine.Locate_Defect(theta);
            % Cutting off defects at edges
            loc12 = temploc12(borderWidth+1:end-borderWidth,borderWidth+1:end-borderWidth);
            loc_12 = temploc_12(borderWidth+1:end-borderWidth,borderWidth+1:end-borderWidth);
            % This next step is to distinguish between empty cells 
            % and cells with defects at x=0 or y=0
            xZ = obj.x(borderWidth+1:end-borderWidth,borderWidth+1:end-borderWidth);
            xZ(xZ == 0) = 0.001;
            yZ = obj.y(borderWidth+1:end-borderWidth,borderWidth+1:end-borderWidth);
            yZ(yZ == 0) = 0.001;
            
            Xtemp12 = nonzeros(xZ.*loc12);
            Ytemp12 = nonzeros(yZ.*loc12);
            Xtemp_12 = nonzeros(xZ.*loc_12);
            Ytemp_12 = nonzeros(yZ.*loc_12);
            
            Xtemp12(Xtemp12 == 0.001) = 0;
            Ytemp12(Ytemp12 == 0.001) = 0;
            Xtemp_12(Xtemp_12 == 0.001) = 0;
            Ytemp_12(Ytemp_12 == 0.001) = 0;
            
            if length(Xtemp12) < nDefects
                Xtemp12 = nan;
                for nInd = 1:nDefects
                    temp_plusDefectTrajX(nInd,j) = nan;
                    temp_plusDefectTrajY(nInd,j) = nan;
                    % temp_minusDefectTrajX(nInd,j) = nan;
                    % temp_minusDefectTrajY(nInd,j) = nan;
                end
            end
            if length(Ytemp12) < nDefects
                Ytemp12 = nan;
                for nInd = 1:nDefects
                    temp_plusDefectTrajX(nInd,j) = nan;
                    temp_plusDefectTrajY(nInd,j) = nan;
                    % temp_minusDefectTrajX(nInd,j) = nan;
                    % temp_minusDefectTrajY(nInd,j) = nan;
                end
            end
            if length(Xtemp_12) < nDefects
                Xtemp_12 = nan;
                for nInd = 1:nDefects
                    % temp_plusDefectTrajX(nInd,j) = nan;
                    % temp_plusDefectTrajY(nInd,j) = nan;
                    temp_minusDefectTrajX(nInd,j) = nan;
                    temp_minusDefectTrajY(nInd,j) = nan;
                end
            end
            if length(Ytemp_12) < nDefects
                Ytemp_12 = nan;
                for nInd = 1:nDefects
                    % temp_plusDefectTrajX(nInd,j) = nan;
                    % temp_plusDefectTrajY(nInd,j) = nan;
                    temp_minusDefectTrajX(nInd,j) = nan;
                    temp_minusDefectTrajY(nInd,j) = nan;
                end
            end
            if sum(isnan(Xtemp12)) + sum(isnan(Ytemp12))  == 0
                for nInd = 1:nDefects
                    temp_plusDefectTrajX(nInd,j) = Xtemp12(nInd);
                    temp_plusDefectTrajY(nInd,j) = Ytemp12(nInd);
                    % temp_minusDefectTrajX(nInd,j) = Xtemp_12(nInd);
                    % temp_minusDefectTrajY(nInd,j) = Ytemp_12(nInd);
                end
            end
            if sum(isnan(Xtemp_12)) + sum(isnan(Ytemp_12)) == 0
                for nInd = 1:nDefects
                    % temp_plusDefectTrajX(nInd,j) = Xtemp12(nInd);
                    % temp_plusDefectTrajY(nInd,j) = Ytemp12(nInd);
                    temp_minusDefectTrajX(nInd,j) = Xtemp_12(nInd);
                    temp_minusDefectTrajY(nInd,j) = Ytemp_12(nInd);
                end
            end

            if mod(j,floor(numFiles/4)) == 0
                disp((j/numFiles)*100 + "% complete");
            end
            
        end
        obj.plusDefectTrajX = temp_plusDefectTrajX;
        % save("plusDefectTrajX.mat","temp_plusDefectTrajX")
        obj.plusDefectTrajY = temp_plusDefectTrajY;
        % save("plusDefectTrajY.mat","temp_plusDefectTrajY")
        obj.minusDefectTrajX = temp_minusDefectTrajX;
        % save("minusDefectTrajX.mat","temp_minusDefectTrajX")
        obj.minusDefectTrajY = temp_minusDefectTrajY;
        % save("minusDefectTrajY.mat","temp_minusDefectTrajY")
        obj.umax = velMax;
        % save("umax.mat","velMax")
        obj.vortmax = vorticityMax;
        % save("vortmax.mat","vorticityMax")
        save

        end

        function plotDirecVort(obj,re)
        ind = (1 + obj.i/obj.outPeriod);
        % Grab Q data, calculate useful fields like u, theta, S, vorticity 
        [Qxx,Qxy] = ATCengine.importQ(ind,obj.directory_name);
        [ux,uy] = ATCengine.vel_field(Qxx,Qxy,obj.kx,obj.ky,obj.k2,obj.alpha,obj.eta,obj.lambda,obj.B,obj.g);
        % S = abs(Qxx + 1i*Qxy);
        theta = 0.5*atan2(Qxy,Qxx);
        vort = ATCengine.vorticity(ux,uy,obj.kx,obj.ky);
        % Calculate defect positions & orientations
        [loc12,loc_12] = ATCengine.Locate_Defect(theta);
        ppol = ATCengine.Polarity(0.5,loc12,Qxx,Qxy,obj.kx,obj.ky);
        npol = ATCengine.Polarity(-0.5,loc_12,Qxx,Qxy,obj.kx,obj.ky);
        % re = 4;
        start_index = 1; 
        end_index = 256;
        % Set the background color to show fluid vorticity
        imagesc(obj.xx+0.5*obj.dx,obj.xx+0.5*obj.dx,vort./obj.vortmax); % max(abs(vort(:)))
        % imagesc(obj.xx,obj.xx,S);
        % imagesc(obj.xx,obj.xx,obj.alpha);
        cbar = colorbar;
        colormap(gca,ATCengine.customColorMaps("blueRed"));
        % caxis([-1 1])
        hold on
        % Plot nematic director; + and - arrows, because of Z2 symmetry
        q = quiver(obj.x(start_index:re:end_index,start_index:re:end_index),obj.y(start_index:re:end_index,start_index:re:end_index),cos(theta(start_index:re:end_index,start_index:re:end_index)),sin(theta(start_index:re:end_index,start_index:re:end_index)),0.5,'.k');
        qm = quiver(obj.x(start_index:re:end_index,start_index:re:end_index),obj.y(start_index:re:end_index,start_index:re:end_index),-cos(theta(start_index:re:end_index,start_index:re:end_index)),-sin(theta(start_index:re:end_index,start_index:re:end_index)),0.5,'.k');
        % Plot positions/orientations of +1/2 defects
        color12 = [0,113,0]./256;
        q12 = quiver(obj.x-0.5*obj.dx,obj.y-0.5*obj.dx,loc12.*cos(ppol),loc12.*sin(ppol),12,'Color',color12,'linewidth',3);
        scatter(obj.x(loc12)-0.5*obj.dx,obj.y(loc12)-0.5*obj.dx,150,color12,'filled');
        % Plot positions/orientations of -1/2 defects
        color_12 = [233,27,150]./256;
        Angle = repelem(npol(loc_12),3,1);
        num_12 = sum(loc_12,'all'); % number of -1/2 defect
        c_x = repelem(obj.x(loc_12)-0.5*obj.dx,3,1); % center of the -1/2 defect repeated 3 times for each vertex of a triangle
        c_y = repelem(obj.y(loc_12)-0.5*obj.dx,3,1);
        modification = (2*pi/3)*repmat([0;1;2],num_12,1); % angle to add to each entry of angle to get 3 vertices
        V = [c_x+(re/4)*cos(Angle+modification) c_y+(re/4)*sin(Angle+modification)]; % modify to adjust triangle size
        F = 3*[1:num_12]';
        F = [F-2 F-1 F];
        patch('Faces',F,'Vertices',V,'FaceColor',color_12)
        scatter(obj.x(loc_12)-0.5*obj.dx,obj.y(loc_12)-0.5*obj.dx,12,color_12,'filled');
        % Plot tweezer contour at 1% maximum magnitude of activity
        alphaThreshold = 0.01*min(min(obj.alpha));
        switch obj.pattern
            case "const"
                disp("Constant activity, no contour");
            case "V-arc"
                [xT,yT] = ATCengine.returnVArcPos(obj.t);
                alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
            case "L-arc"
                [xT,yT] = ATCengine.returnLArcPos(obj.t);
                alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
            case "angBraid"
                alphaThreshold = -1;
                braidPattern = ATCengine.returnSimplifiedBraidActivity(obj.x,obj.y,obj.t);
                alphaContour = contour(obj.x,obj.y,braidPattern,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
            otherwise
                alphaThreshold = 0.01*min(min(obj.alpha));
                alphaContour = contour(obj.x,obj.y,obj.alpha,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
        end
        % if obj.pattern == "L-arc" || obj.pattern == "V-arc" || obj.pattern == "angBraid"
        %     switch obj.pattern
        % 
        %         case "V-arc"
        %             [xT,yT] = ATCengine.returnVArcPos(obj.t);
        %         case "L-arc"
        %             [xT,yT] = ATCengine.returnLArcPos(obj.t);
        %         case "angBraid"
        %     end
        %     alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
        % elseif obj.pattern ~= "const" 
        %     if obj.pattern == "angBraid"
        %         alphaThreshold = -0.1;
        %     else
        %         alphaThreshold = 0.01*min(min(obj.alpha));
        %     end
        %     alphaContour = contour(obj.x,obj.y,obj.alpha,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
        % end
        hold off;
        % Set view limits
        xlim([min(obj.xx) max(obj.xx)]); 
        ylim([min(obj.xx) max(obj.xx)]);
        % pbaspect([obj.gridLength obj.gridLength 1])
        xlabel("X",'FontSize',22)
        ylabel("Y",'FontSize',22)
        ylabel(cbar, " \hspace{2pt} $\frac{\omega}{\omega_{max}}$",'FontSize',38,'Interpreter','latex','Rotation',0,'VerticalAlignment','middle')
        clim([-1 1]);
        ax = gca;
        ax.FontSize = 22;
        xticks(-60:30:60);
        yticks(-60:30:60);
        view(0,-90)
        end

        function plotFluidVel(obj,re)
        ind = (1 + obj.i/obj.outPeriod);
        % Grab Q data, calculate useful fields like u, theta, S, vorticity 
        [Qxx,Qxy] = ATCengine.importQ(ind,obj.directory_name);
        [ux,uy] = ATCengine.vel_field(Qxx,Qxy,obj.kx,obj.ky,obj.k2,obj.alpha,obj.eta,obj.lambda,obj.B,obj.g);
        u = (ux.^2 + uy.^2).^0.5;
        % S = abs(Qxx + 1i*Qxy);
        theta = 0.5*atan2(Qxy,Qxx);
        % vort = ATCengine.vorticity(ux,uy,obj.kx,obj.ky);
        % Calculate defect positions & orientations
        [loc12,loc_12] = ATCengine.Locate_Defect(theta);
        ppol = ATCengine.Polarity(0.5,loc12,Qxx,Qxy,obj.kx,obj.ky);
        npol = ATCengine.Polarity(-0.5,loc_12,Qxx,Qxy,obj.kx,obj.ky);
        % re = 4;
        start_index = 1; 
        end_index = 256;
        % Set the background color to show fluid flow speed
        imagesc(obj.xx+0.5*obj.dx,obj.xx+0.5*obj.dx,u./obj.umax);
        % imagesc(obj.xx,obj.xx,S);
        % imagesc(obj.xx,obj.xx,obj.alpha);
        cbar = colorbar;
        colormap(gca,ATCengine.customColorMaps("blueGray"));
        hold on
        % Plot fluid flow
        q = quiver(obj.x(start_index:re:end_index,start_index:re:end_index),obj.y(start_index:re:end_index,start_index:re:end_index),ux(start_index:re:end_index,start_index:re:end_index),uy(start_index:re:end_index,start_index:re:end_index),1.5,'k');
        % Plot positions/orientations of +1/2 defects
        color12 = [0,113,0]./256;
        q12 = quiver(obj.x-0.5*obj.dx,obj.y-0.5*obj.dx,loc12.*cos(ppol),loc12.*sin(ppol),12,'Color',color12,'linewidth',3);
        scatter(obj.x(loc12)-0.5*obj.dx,obj.y(loc12)-0.5*obj.dx,150,color12,'filled');
        % Plot positions/orientations of -1/2 defects
        color_12 = [233,27,150]./256;
        Angle = repelem(npol(loc_12),3,1);
        num_12 = sum(loc_12,'all'); % number of -1/2 defect
        c_x = repelem(obj.x(loc_12)-0.5*obj.dx,3,1); % center of the -1/2 defect repeated 3 times for each vertex of a triangle
        c_y = repelem(obj.y(loc_12)-0.5*obj.dx,3,1);
        modification = (2*pi/3)*repmat([0;1;2],num_12,1); % angle to add to each entry of angle to get 3 vertices
        V = [c_x+(re/4)*cos(Angle+modification) c_y+(re/4)*sin(Angle+modification)]; % modify to adjust triangle size
        F = 3*[1:num_12]';
        F = [F-2 F-1 F];
        patch('Faces',F,'Vertices',V,'FaceColor',color_12)
        scatter(obj.x(loc_12)-0.5*obj.dx,obj.y(loc_12)-0.5*obj.dx,12,color_12,'filled');
        % Plot tweezer contour at 1% maximum magnitude of activity
        alphaThreshold = 0.01*min(min(obj.alpha));
        switch obj.pattern
            case "const"
                disp("Constant activity, no contour");
            case "V-arc"
                [xT,yT] = ATCengine.returnVArcPos(obj.t);
                alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
            case "L-arc"
                [xT,yT] = ATCengine.returnLArcPos(obj.t);
                alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
            case "angBraid"
                alphaThreshold = -1;
                braidPattern = ATCengine.returnSimplifiedBraidActivity(obj.x,obj.y,obj.t);
                alphaContour = contour(obj.x,obj.y,braidPattern,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
            otherwise
                alphaThreshold = 0.01*min(min(obj.alpha));
                alphaContour = contour(obj.x,obj.y,obj.alpha,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
        end
        % if obj.pattern == "L-arc" || obj.pattern == "V-arc"
        %     switch obj.pattern
        %         case "V-arc"
        %             [xT,yT] = ATCengine.returnVArcPos(obj.t);
        %         case "L-arc"
        %             [xT,yT] = ATCengine.returnLArcPos(obj.t);
        %     end
        %     alphaContour = contour(obj.x,obj.y,tanh( sqrt((obj.x-xT).^2 + (obj.y-yT).^2) - obj.R),[0,0],'--k','LineWidth',2);
        % elseif obj.pattern ~= "const" 
        %     if obj.pattern == "angBraid"
        %         alphaThreshold = -0.1;
        %     else
        %         alphaThreshold = 0.01*min(min(obj.alpha));
        %     end
        %     alphaContour = contour(obj.x,obj.y,obj.alpha,[alphaThreshold,alphaThreshold],'--k','LineWidth',2);
        % end
        hold off;
        % Set view limits
        xlim([min(obj.xx) max(obj.xx)]); 
        ylim([min(obj.xx) max(obj.xx)]);
        % pbaspect([obj.gridLength obj.gridLength 1])
        xlabel("X",'FontSize',22)
        if obj.pattern ~= "angBraid"
            ylabel("Y",'FontSize',22)
        end
        ylabel(cbar, " \hspace{12pt} $\frac{u}{u_{max}}$",'FontSize',38,'Interpreter','latex','Rotation',0,'VerticalAlignment','middle')
        clim([0 1]);
        ax = gca;
        ax.FontSize = 22;
        xticks(-60:30:60);
        yticks(-60:30:60);
        view(0,-90)
        end

        function plotCMActivity(obj)
        centeredAlpha = obj.alpha;
        alphaThreshold = 0.01*min(min(centeredAlpha));
        imagesc(obj.xx,obj.xx,-1.*centeredAlpha);
        colormap(gca,hot);
        hold on
        if obj.pattern == "L-arc" || obj.pattern == "V-arc"
            contour(obj.x,obj.y,tanh( sqrt((obj.x).^2 + (obj.y).^2) - obj.R),[0,0],'--w','LineWidth',2);
        elseif obj.pattern ~= "const" 
            contour(obj.x,obj.y,centeredAlpha,[alphaThreshold,alphaThreshold],'--w','LineWidth',2);
        end
        axis equal;
        xlim([-1.15*obj.R, 1.15*obj.R]);
        ylim([-1.15*obj.R, 1.15*obj.R]);
        xticks(-10:5:10);
        yticks(-10:5:10);
        ax = gca;
        ax.FontSize = 16;
        cbarAlpha = colorbar;
        upperLim = -1.1*obj.a0;
        lowerLim = min([0,min(min(-1*centeredAlpha))]);
        if obj.pattern ~= "const" 
            clim([lowerLim,upperLim]);        
        end
        xlabel("X_{CM}",'FontSize',16)
        ylabel("Y_{CM}",'FontSize',16)
        ylabel(cbarAlpha, " \hspace{15pt} $|\alpha|$",'FontSize',28,'Interpreter','latex','Rotation',0,'VerticalAlignment','middle')% ylabel(cbar, "\alpha",'FontSize',25)
        title("Activity Pattern",'FontSize',16,'Interpreter','latex')
        view(0,-90)
        end
    
        function plotDefectTrajs(obj)
        j = 1 + obj.i/obj.outPeriod;
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;
        tspan = (0:(numFiles-1)).*(obj.outPeriod*obj.dt);
        colorArray12 = 0.25*(tspan./obj.T);
        colorArray_12 = 1 - 0.25*(tspan./obj.T);
        scatter3(obj.plusDefectTrajX(1:j),obj.plusDefectTrajY(1:j),tspan(1:j),50,fliplr(colorArray12(1:j)),'filled');
        hold on
        scatter3(obj.minusDefectTrajX(1:j),obj.minusDefectTrajY(1:j),tspan(1:j),50,fliplr(colorArray_12(1:j)),'filled');
        colormap(gca,ATCengine.customColorMaps('defects'));
        % % Averaged tweezer flow velocity
        % ind = obj.i / obj.outPeriod;
        % [Qxx,Qxy] = ATCengine.importQ(ind,obj.directory_name);
        % [ux,uy] = ATCengine.vel_field(Qxx,Qxy,obj.kx,obj.ky,obj.k2,obj.alpha,obj.eta,obj.lambda,obj.B,obj.g);
        % switch obj.pattern
        %     case "L-arc"
        %         [xT,yT] = ATCengine.returnLArcPos(obj.t);
        %         vT = 0.05;
        %     case "V-arc"
        %         [xT,yT] = ATCengine.returnVArcPos(obj.t);
        %         vT = 0.025*sqrt(2);
        %     otherwise
        %         xT = obj.x0;
        %         yT = obj.y0;
        % end
        % rT = 6;
        % [Vx,Vy] = ATCengine.computeMeanFlow(obj.x,obj.y,ux,uy,xT,yT,rT,obj.umax);
        % flowDirection = quiver3(xT,yT,0,Vx,Vy,0,50,'color','k','LineWidth',1.5,'marker','x');
        % viscircles([xT,yT], rT,'LineStyle','--');
        axis equal;
        grid on;
        xlim([min(obj.xx) max(obj.xx)])
        ylim([min(obj.xx) max(obj.xx)])
        ax = gca;
        ax.FontSize = 16;
        xticks(-60:30:60);
        yticks(-60:30:60);
        xlabel("X",'FontSize',16)
        ylabel("Y",'FontSize',16)
        title("Defect Trajectories",'FontSize',16,'Interpreter','latex')
        set(gca, 'YDir','reverse')
        view(0,-90)
        end
        
        function plotBraidTrajs(obj)
        j = (1 + obj.i/obj.outPeriod);
        files = dir(obj.directory_name+"/*.dat");
        numFiles = size(files,1) - 1;
        tspan = (obj.outPeriod*obj.dt)*(0:numFiles);
        colorArray12 = 0.25*(tspan./obj.T);
        colorArray_12 = 1 - 0.25*(tspan./obj.T);
        hold on
        j = max([1,j]);
        for nInd = [1,2]
            scatter3(obj.plusDefectTrajX(nInd,1:j),obj.plusDefectTrajY(nInd,1:j),tspan(1:j),50,fliplr(colorArray12(1:j)),'filled');
            scatter3(obj.minusDefectTrajX(nInd,1:j),obj.minusDefectTrajY(nInd,1:j),tspan(1:j),50,fliplr(colorArray_12(1:j)),'filled');
        end
        view(5+(360*obj.t/obj.T),10)
        % view(5,10)
        colormap(gca,ATCengine.customColorMaps('defects'));
        % % Averaged tweezer flow velocity
        % ind = obj.i / obj.outPeriod;
        % [Qxx,Qxy] = ATCengine.importQ(ind,obj.directory_name);
        % [ux,uy] = ATCengine.vel_field(Qxx,Qxy,obj.kx,obj.ky,obj.k2,obj.alpha,obj.eta,obj.lambda,obj.B,obj.g);
        % switch obj.pattern
        %     case "L-arc"
        %         [xT,yT] = ATCengine.returnLArcPos(obj.t);
        %         vT = 0.05;
        %     case "V-arc"
        %         [xT,yT] = ATCengine.returnVArcPos(obj.t);
        %         vT = 0.025*sqrt(2);
        %     otherwise
        %         xT = obj.x0;
        %         yT = obj.y0;
        % end
        % rT = 6;
        % [Vx,Vy] = ATCengine.computeMeanFlow(obj.x,obj.y,ux,uy,xT,yT,rT,obj.umax);
        % flowDirection = quiver3(xT,yT,0,Vx,Vy,0,50,'color','k','LineWidth',1.5,'marker','x');
        % viscircles([xT,yT], rT,'LineStyle','--');
        ax = gca;
        ax.FontSize = 16;
        grid on
        xticks(-60:60:60);
        yticks(-60:60:60);
        xlabel("X",'FontSize',16)
        ylabel("Y",'FontSize',16)
        zlabel("Time",'FontSize',16,'Interpreter','latex')
        title("Defect Worldlines",'FontSize',16,'Interpreter','latex')
        end
    
    end

    methods(Static)
        
        %====================================================================================================================================================
        % MISC STATIC FUNCTIONS
        %====================================================================================================================================================
        function k = wavenums(xx)
        n = length(xx);
        high = floor(n/2);
        low = -floor((n-1)/2);
        L = n*(xx(2)-xx(1));
        k = [(0:high) (low:-1)]*2*pi/L;
        end 
        
        function Q2 = TraceQ(Qxx,Qxy)
        Q2 = 2*Qxx.^2 + 2*Qxy.^2;
        end
        
        function [Qxx,Qxy] = Q_constructor_2D(S,nx,ny)
        Qxx = S.*(nx .* nx - 1/2);
        Qxy = S.*nx .* ny;
        %Qyx = Qxy and Qyy = -Qxx
        end

        function [Qxx,Qxy] = setUnifQ(k2,angle)
        if nargin < 2
            angle = 0.5*pi;
        end
        theta = angle*ones(size(k2)) + 0.01*pi*rand(size(k2));
        S = 0.95*ones(size(k2)) + 0.05*rand(size(k2));
        nx = cos(theta); 
        ny= sin(theta);
        [Qxx,Qxy] = ATCengine.Q_constructor_2D(S,nx,ny);
        end

        function [Qxx,Qxy] = setrandQ(k2,B,dt)
        theta = 0.5*pi*ones(size(k2)) + pi*rand(size(k2));
        S = 0.95*ones(size(k2)) + 0.05*rand(size(k2));
        nx = cos(theta); 
        ny= sin(theta);
        [Qxx,Qxy] = ATCengine.Q_constructor_2D(S,nx,ny);
        for m = 1:100
            [Qxx,Qxy] = ATCengine.relaxQ(Qxx,Qxy,k2,B,dt);
        end
        end

        function [Qxx,Qxy] = setSingleDefect(x,y,charge,xpos,ypos,phi0)
        theta = charge*atan2(y-ypos,x-xpos) + phi0 + 0.01*pi*rand(size(x));
        S = 0.95*ones(size(x)) + 0.05*rand(size(x));
        nx = cos(theta); 
        ny= sin(theta);
        [Qxx,Qxy] = ATCengine.Q_constructor_2D(S,nx,ny);
        end

        function exportQ(Qxx,Qxy,t,directory_name)
        outQxx = reshape(Qxx,[numel(Qxx),1]);
        outQxy = reshape(Qxy,[numel(Qxy),1]);
        outQ = [outQxx,outQxy];
        QFilename = [convertStringsToChars(directory_name),'/Q_conf_',num2str(t),'.dat'];
        save(QFilename,'outQ','-ascii');
        end
        
        function sim = importSimData(directory_name_string)
        directory_name = char(directory_name_string);
        sim_file_name = [directory_name,'/simData.mat'];
        simData = load(sim_file_name);
        sim = simData.obj;
        end

        function [Qxx,Qxy] = importQ(t,directory_name_string,Nx,Ny)
        if nargin < 2
            directory_name = 'ATCdata';
        else
            directory_name = char(directory_name_string);
        end
        QFilename = [directory_name,'/Q_conf_',num2str(t),'.dat'];
        Qdata = load(QFilename);
        if nargin < 3
            Nx = sqrt(numel(Qdata(:,1)));
            Ny = sqrt(numel(Qdata(:,1)));
        end
        Qxx = reshape(Qdata(:,1),[Ny,Nx]);
        Qxy = reshape(Qdata(:,2),[Ny,Nx]);
        end

        function [ux,uy] = vel_field(Qxx,Qxy,kx,ky,k2,alpha,eta,lambda,B,g)
        Q2 = ATCengine.TraceQ(Qxx,Qxy);
        [uaktx, uakty] = ATCengine.active_stress(Qxx,Qxy,kx,ky,alpha);
        % uelx = zeros(size(uaktx));
        % uely = zeros(size(uaktx));
        % px = zeros(size(uaktx));
        % py = zeros(size(uaktx));
        % uy = uakty + uely - eta*real(ifft2(k2 .* fft2(uy)));
        % ufx = fft2(ux);
        % ufy = fft2(uy);
        % Comment out below to remove elastic stress and incompressibility
        [Hxx,Hxy] = ATCengine.H_const(Qxx,Qxy,Q2,k2,B);
        [PI_xx,PI_xy,PI_yx,PI_yy]= ATCengine.elastic_stress(Qxx,Qxy,Hxx,Hxy,kx,ky,lambda);
        [uelx, uely] = ATCengine.vel_elastic_stress(PI_xx,PI_xy,PI_yx,PI_yy,kx,ky,g);
        [px,py] = ATCengine.pressure(Qxx,Qxy,PI_xx,PI_xy,PI_yx,PI_yy,alpha,g,kx,ky,k2);
        ufx = ( fft2(uaktx) + fft2(uelx) - fft2(px) )./(1+eta*k2);
        ufy = ( fft2(uakty) + fft2(uely) - fft2(py) )./(1+eta*k2);
        
        ux = real(ifft2(ufx));
        uy = real(ifft2(ufy));
        end
        
        function omega = vorticity(ux,uy,kx,ky)
        uxf = fft2(ux);
        uyf = fft2(uy);
        omega = real(ifft2(1i*kx.*uyf)) -real(ifft2(1i*ky.*uxf));
        end

        function [Qxx,Qxy] = flowEvolveQ(Qxx,Qxy,kx,ky,k2,alpha,eta,lambda,B,g,dt)
        Q2 = ATCengine.TraceQ(Qxx,Qxy);
        [ux,uy] = ATCengine.vel_field(Qxx,Qxy,kx,ky,k2,alpha,eta,lambda,B,g);
        [Omega_xy, Exx,Exy ] =  ATCengine.Vorticity_and_strain(ux,uy,kx,ky);
        [advecQxx,advecQxy] = ATCengine.advecQ_func(Qxx,Qxy,ux,uy,kx,ky);
        small_omega = 1-k2;
        integ = exp(small_omega*dt);
        
        
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        [N0xx,N0xy] = ATCengine.Nf(Qxx,Qxy,advecQxx,advecQxy,Q2,Omega_xy,Exx,Exy,lambda,B);
        N0xx = fft2(N0xx);
        N0xy = fft2(N0xy);
        
        Qfxx = Qfxx.*integ + N0xx .*(integ -1)./small_omega;
        Qfxy = Qfxy.*integ + N0xy .*(integ -1)./small_omega;
        
        Qxx = real(ifft2(Qfxx));
        Qxy = real(ifft2(Qfxy));
        
        
        Q2 = ATCengine.TraceQ(Qxx,Qxy);
        [ux,uy] = ATCengine.vel_field(Qxx,Qxy,kx,ky,k2,alpha,eta,lambda,B,g);
        [Omega_xy, Exx,Exy ] =  ATCengine.Vorticity_and_strain(ux,uy,kx,ky);
        [advecQxx,advecQxy] = ATCengine.advecQ_func(Qxx,Qxy,ux,uy,kx,ky);
        
        
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        [N1xx,N1xy] = ATCengine.Nf(Qxx,Qxy,advecQxx,advecQxy,Q2,Omega_xy,Exx,Exy,lambda,B);
        N1xx = fft2(N1xx) - N0xx;
        N1xy = fft2(N1xy) - N0xy;
        
        Qfxx = Qfxx + N1xx .*((integ -1)./(small_omega*dt) -1)./small_omega;
        Qfxy = Qfxy + N1xy .*((integ -1)./(small_omega*dt) -1)./small_omega;
        
        Qxx = real(ifft2(Qfxx));
        Qxy = real(ifft2(Qfxy));
        
        end
        
        function [uaktx,uakty] = active_stress(Qxx,Qxy,kx,ky,alpha)
        % Actually the divergence of the active stress
        uaktx =  real(ifft2(1i*kx.*fft2(alpha.*Qxx) + 1i*ky .* fft2(alpha.*Qxy)));
        uakty =  real(ifft2(1i*kx.*fft2(alpha.*Qxy) - 1i*ky .* fft2(alpha.*Qxx)));
        end
        
        function [Hxx,Hxy] = H_const(Qxx,Qxy,Q2,k2,B)
        factor = 1 - 2*Q2/B;
        Hxx = factor.*Qxx + real(ifft2(-k2.*fft2(Qxx)));
        Hxy = factor.*Qxy + real(ifft2(-k2.*fft2(Qxy)));
        %H has the same symmetries as Q
        %This means that it is symmetric and traceless
        end
        
        function [PI_xx,PI_xy,PI_yx,PI_yy]= elastic_stress(Qxx,Qxy,Hxx,Hxy,kx,ky,lambda)
        
        %k2 = kx.^2 + ky.^2;
        %Hxx =  real(ifft2(-k2.*fft2(Qxx)));
        %Hxy = real(ifft2(-k2.*fft2(Qxx)));
        
        TraceQH = 2*Qxx.*Hxx + 2*Qxy.*Hxy;
        
        %We are first going to sum up the antisymmetric part
        % That is the Qik*Hkj - Hik*Qkj
        Anty_xy = 2*Qxx.*Hxy -2*Qxy.*Hxx;
        %Anty_yx = -Anty_xy;
        
        %the rest Is the  symmetric traseless and the differential part:
        % %Previous incorrect version
        % Symm_Trace_xx = lambda*TraceQH.*Qxx +lambda*TraceQH -lambda*Hxx;
        
        % The following was checked and confirmed to be correct by Luca, 6/21/22
        Symm_Trace_xx = 2*lambda*TraceQH.*Qxx -lambda*Hxx;
        Symm_Trace_xx = Symm_Trace_xx - 2*lambda*Hxx.*Qxx - 2*lambda*Hxy.*Qxy;
        Symm_Trace_xy = lambda*TraceQH.*Qxy -lambda*Hxy;
        %The Hxk*Qky + Qxk*Hky = 0
        
        %The last part is the differential term, This is symmetric, 
        %But not traceless
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        
        Diff_xx = -2* real(ifft2(1i*kx.*Qfxx)).^2 - 2* real(ifft2(1i*kx.*Qfxy)).^2;
        Diff_yy = -2* real(ifft2(1i*ky.*Qfxx)).^2 - 2* real(ifft2(1i*ky.*Qfxy)).^2;
        
        Diff_xy = -2*real(ifft2(1i*kx.*Qfxx)).*real(ifft2(1i*ky.*Qfxx));
        Diff_xy = Diff_xy -2*real(ifft2(1i*kx.*Qfxy)).*real(ifft2(1i*ky.*Qfxy));
        
        %Here we constrict the Stress
        
        PI_xx = Symm_Trace_xx + Diff_xx; 
        PI_xy = Symm_Trace_xy + Diff_xy + Anty_xy;
        PI_yx = Symm_Trace_xy + Diff_xy - Anty_xy;
        PI_yy = -Symm_Trace_xx + Diff_yy;
        
        end
        
        function [uelx, uely] = vel_elastic_stress(PI_xx,PI_xy,PI_yx,PI_yy,kx,ky,g)
        %[Hxx,Hxy] = H_const(Qxx,Qxy,Q2,k2,B);
        %[PI_xx,PI_xy,PI_yx,PI_yy] = elastic_stress(Qxx,Qxy,Hxx,Hxy,kx,ky,lambda);
        uelx = g*(real(ifft2(1i*kx.*fft2(PI_xx))) + real(ifft2(1i*ky.*fft2(PI_yx))));
        uely = g*(real(ifft2(1i*kx.*fft2(PI_xy))) + real(ifft2(1i*ky.*fft2(PI_yy))));
        end
        
        function [px, py] = pressure(Qxx,Qxy,PI_xx,PI_xy,PI_yx,PI_yy,alpha,g,kx,ky,k2)
        %Removing singularity
        k2(1)= 1;
        %From activity
        pf = ((kx.^2 - ky.^2).*fft2(alpha.*Qxx) +2*kx.*ky.*fft2(alpha.*Qxy));
        %From elasticity
        pf = pf + g *(kx.^2.*fft2(PI_xx) + kx.*ky.*(fft2(PI_xy) + fft2(PI_yx)) +ky.^2 .*fft2(PI_yy));
        pf = pf ./k2;
        
        px = real(ifft2(1i*kx.*pf));
        py = real(ifft2(1i*ky.*pf));
        
        end
        
        function [Omega_xy, Exx,Exy] = Vorticity_and_strain(ux,uy,kx,ky)
        uxf = fft2(ux); uyf = fft2(uy);
        
        dxux = real(ifft2(1i *kx .* uxf));
        dxuy = real(ifft2(1i *kx .* uyf));
        dyux = real(ifft2(1i *ky .* uxf));
        dyuy = real(ifft2(1i *ky .* uyf));
        
        %div = dxux + dyuy;
        
        %Even thougt Omega is antisymetric i keep two components to eas the work
        Omega_xy = 0.5*(dxuy - dyux);
        %Omega_yx = -Omega_xy; 
        
        Exx = 0.5*(dxux - dyuy);
        Exy = 0.5*(dxuy + dyux);
        
        %E is symetric and traseless
        %Eyx = Exy and Eyy = -Exx
        end
        
        function [advecQxx, advecQxy] = advecQ_func(Qxx,Qxy,ux,uy,kx,ky)
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        advecQxx = ux.*real(ifft2(1i*kx .* Qfxx)) +uy.*real(ifft2(1i*ky .* Qfxx));
        advecQxy = ux.*real(ifft2(1i*kx .* Qfxy)) +uy.*real(ifft2(1i*ky .* Qfxy));
        end
        
        function [Nxx,Nxy] = Nf(Qxx,Qxy,advecQxx,advecQxy,Q2,Omega_xy,Exx,Exy,lambda,B)
        [Wxx,Wxy] = ATCengine.W(Qxx,Qxy,Omega_xy,Exx,Exy,lambda);
        Nxx = Wxx -advecQxx - 2*Q2 .*Qxx/B; 
        Nxy = Wxy -advecQxy - 2*Q2 .*Qxy/B;
        end
        
        function [Wxx,Wxy] = W(Qxx,Qxy,Omega_xy,Exx,Exy,lambda)
        TraceEQ = 2*Qxx.*Exx +2* Qxy.*Exy; 
        
        Wxx = -2*Omega_xy.*Qxy + lambda*Exx + 2*lambda*(Qxx.*Exx + Qxy.*Exy);
        Wxx = Wxx - lambda *( TraceEQ.*Qxx + TraceEQ);
        
        Wxy = 2*Qxx.*Omega_xy + lambda*Exy -lambda*TraceEQ.*Qxy;
        
        %The Exk*Qky + Qxk*Eky =0
        
        end

        function [Qxx,Qxy] = relaxQ(Qxx,Qxy,k2,B,dt)
        Q2 = ATCengine.TraceQ(Qxx,Qxy);
        small_omega = 1-k2;
        integ = exp(small_omega*dt);
        
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        [N0xx, N0xy] = ATCengine.N_NO_flow(Qxx,Qxy,Q2,B);
        N0xx = fft2(N0xx);
        N0xy = fft2(N0xy);
        
        Qfxx = Qfxx.*integ + N0xx .*(integ -1)./small_omega;
        Qfxy = Qfxy.*integ + N0xy .*(integ -1)./small_omega;
        
        Qxx = real(ifft2(Qfxx));
        Qxy = real(ifft2(Qfxy));
        
        %N0(:,:,1,j) = N0ij;
        
        Q2 = ATCengine.TraceQ(Qxx,Qxy);
        
        Qfxx = fft2(Qxx);
        Qfxy = fft2(Qxy);
        [N1xx,N1xy] = ATCengine.N_NO_flow(Qxx,Qxy,Q2,B);
        N1xx = fft2(N1xx) - N0xx;
        N1xy = fft2(N1xy) - N0xy;
        
        Qfxx = Qfxx + N1xx .*((integ -1)./(small_omega*dt) -1)./small_omega;
        Qfxy = Qfxy + N1xy .*((integ -1)./(small_omega*dt) -1)./small_omega;
        Qxx = real(ifft2(Qfxx));
        Qxy = real(ifft2(Qfxy));
        
        
        end

        function [Nxx,Nxy] = N_NO_flow(Qxx,Qxy,Q2,B)
        Nxx = - 2*Q2 .*Qxx/B;
        Nxy = - 2*Q2 .*Qxy/B;
        end
        
        function [loc12,loc_12] = Locate_Defect(theta)
        %{
        Consider the following matrix where each entry represent angle of the
        particle. 
        theta = [ A | B ]
                [ C | D ]
        
        then we can calculate angle around a point D by using circshift
        theta_left = [ B | A ]
                     [ D | C ]
                     --------> shifted right so that element (i,j) of theta_left
                               represents a grid to the left of original theta
        
        theta_up   = [ C | D ] |  shifted down so that element (i,j)
                     [ A | B ] v  represents a grid to the above of original theta 
        %}
        theta = mod(theta,pi);
        theta_left = circshift(theta,[0 1]);
        theta_up = circshift(theta,[1 0]);
        theta_left_up = circshift(theta,[1 1]);
        % We then use this to calculate the difference between theta of a
        % loop in space
        up_diff =      ATCengine.diffCCW(theta,theta_up);
        up_left_diff = ATCengine.diffCCW(theta_up,theta_left_up);
        left_diff =    ATCengine.diffCCW(theta_left_up,theta_left);
        diff =         ATCengine.diffCCW(theta_left,theta);
        % Then we check if angle is pi/2 or -pi/2 or 0.
        tol = 0.3;
        strength = up_diff+up_left_diff+left_diff+diff;
        loc12 = abs(strength+pi)/pi < tol;
        loc_12 = abs(strength-pi)/pi < tol;

        end
        
        function delta = diffCCW(a,b)
        c = b-a;
        % There are 3 cases
        % First C is greater then pi/2 then we have to invert angle of b to get the
        % smallest difference
        c(c >= pi/2) = c(c >= pi/2) - pi;
        % Second case is when C is less then -pi/2. We have to invert angle of a
        c(c <= -pi/2) = c(c <= -pi/2) + pi;
        delta = c;
        end
        
        function pol = Polarity(q,loc,Qxx,Qxy,kx,ky)        
            Qfxx = fft2(Qxx);
            Qfxy = fft2(Qxy);
            dxQxx = real(ifft2(1i*kx.*Qfxx));
            dyQxx = real(ifft2(1i*ky.*Qfxx));
            dxQxy = real(ifft2(1i*kx.*Qfxy));
            dyQxy = real(ifft2(1i*ky.*Qfxy));
            
            numerator = sign(q)*(dxQxy)-dyQxx; % This is the numerator 
            % of eq.5 in Orientational properties of nematic disclinations by 
            % Vromans and Giomi.
            denominator = dxQxx+sign(q)*(dyQxy);
            av_num = ATCengine.av_loop(numerator);
            av_denum = ATCengine.av_loop(denominator);
            
            pol = q/(1-q)*loc.*atan2(av_num,av_denum);
        end

        function av = av_loop(x)
            %{
            We are following a similar algorithm from Detect_Defect.m 
            Consider the following matrix x on a grid.
            x  =    [ A | B ]
                    [ C | D ]
        
            then we can calculate angle around a point D by using circshift
            x_left  =    [ B | A ]
                         [ D | C ]
                         --------> shifted right so that element (i,j)
                                   represents a grid to the left of original theta
        
            x_up    =    [ C | D ] |  shifted down so that element (i,j)
                         [ A | B ] v  represents a grid to the above of original theta 
        
            %}
            
            x_left = circshift(x,[0 1]);
            x_up = circshift(x,[1 0]);
            x_left_up = circshift(x,[1 1]);
            
            % Then we calculate the average
            av = (x_left+x_up+x_left_up+x)/4;
        end
        
        function map = customColorMaps(tag,n)
        if nargin < 2
            n = 128;
        end
        if nargin < 1
            tag = "blueRed";
        end
        switch tag
            case "blueRed"
                baseMap = [175*0.7  53*0.7   71*0.7 ;
                           216*0.8   82*0.8   88*0.8 ;
                           239*0.9  133*0.9  122*0.9 ;
                           245 177 139;
                           249 216 168;
                           242 238 197;
                           216 236 241;
                           154 217 238;
                            68*0.9  199*0.9  239*0.9 ;
                             0*0.8  170*0.8  226*0.8 ;
                             0*0.7  116*0.7  188*0.7 ]/255;
                idx1 = linspace(0,1,size(baseMap,1));
                idx2 = linspace(0,1,n);
                map = interp1(idx1,baseMap,idx2);
                map = flipud(map);
            case "blueGray"
                baseMap = [  0*0.8 170*0.8 255*0.8;
                            53*0.8 196*0.8 243*0.8;
                           133*0.8 212*0.8 234*0.8;
                           190 230 232;
                           217 224 230]/255;
                idx1 = linspace(0,1,size(baseMap,1));
                idx2 = linspace(0,1,n);
                map = interp1(idx1,baseMap,idx2);
                map = flipud(map);
            case "defects"
                map = [ 0.1529    0.3922    0.0941
                        0.1589    0.3994    0.0951
                        0.1649    0.4066    0.0960
                        0.1708    0.4138    0.0969
                        0.1768    0.4210    0.0979
                        0.1827    0.4282    0.0988
                        0.1887    0.4355    0.0998
                        0.1947    0.4427    0.1007
                        0.2006    0.4499    0.1016
                        0.2066    0.4571    0.1026
                        0.2125    0.4643    0.1035
                        0.2185    0.4715    0.1045
                        0.2245    0.4787    0.1054
                        0.2304    0.4860    0.1064
                        0.2364    0.4932    0.1073
                        0.2424    0.5004    0.1082
                        0.2483    0.5076    0.1092
                        0.2543    0.5148    0.1101
                        0.2602    0.5220    0.1111
                        0.2662    0.5293    0.1120
                        0.2722    0.5365    0.1129
                        0.2781    0.5437    0.1139
                        0.2841    0.5509    0.1148
                        0.2900    0.5581    0.1158
                        0.2960    0.5653    0.1167
                        0.3020    0.5725    0.1176
                        0.3095    0.5789    0.1231
                        0.3170    0.5852    0.1285
                        0.3246    0.5916    0.1339
                        0.3321    0.5979    0.1394
                        0.3397    0.6042    0.1448
                        0.3472    0.6106    0.1502
                        0.3548    0.6169    0.1557
                        0.3623    0.6232    0.1611
                        0.3698    0.6296    0.1665
                        0.3774    0.6359    0.1719
                        0.3849    0.6422    0.1774
                        0.3925    0.6486    0.1828
                        0.4000    0.6549    0.1882
                        0.4075    0.6612    0.1937
                        0.4151    0.6676    0.1991
                        0.4226    0.6739    0.2045
                        0.4302    0.6802    0.2100
                        0.4377    0.6866    0.2154
                        0.4452    0.6929    0.2208
                        0.4528    0.6992    0.2262
                        0.4603    0.7056    0.2317
                        0.4679    0.7119    0.2371
                        0.4754    0.7183    0.2425
                        0.4830    0.7246    0.2480
                        0.4905    0.7309    0.2534
                        0.4980    0.7373    0.2588
                        0.5071    0.7429    0.2692
                        0.5162    0.7485    0.2795
                        0.5253    0.7542    0.2899
                        0.5344    0.7598    0.3002
                        0.5435    0.7655    0.3106
                        0.5526    0.7711    0.3209
                        0.5617    0.7768    0.3313
                        0.5708    0.7824    0.3416
                        0.5799    0.7881    0.3520
                        0.5890    0.7937    0.3624
                        0.5981    0.7994    0.3727
                        0.6072    0.8050    0.3831
                        0.6163    0.8107    0.3934
                        0.6254    0.8163    0.4038
                        0.6345    0.8220    0.4141
                        0.6436    0.8276    0.4245
                        0.6527    0.8333    0.4348
                        0.6618    0.8389    0.4452
                        0.6709    0.8445    0.4555
                        0.6800    0.8502    0.4659
                        0.6891    0.8558    0.4762
                        0.6982    0.8615    0.4866
                        0.7073    0.8671    0.4969
                        0.7164    0.8728    0.5073
                        0.7255    0.8784    0.5176
                        0.7321    0.8814    0.5305
                        0.7388    0.8845    0.5433
                        0.7454    0.8875    0.5561
                        0.7520    0.8905    0.5689
                        0.7587    0.8935    0.5817
                        0.7653    0.8965    0.5946
                        0.7719    0.8995    0.6074
                        0.7786    0.9026    0.6202
                        0.7852    0.9056    0.6330
                        0.7919    0.9086    0.6459
                        0.7985    0.9116    0.6587
                        0.8051    0.9146    0.6715
                        0.8118    0.9176    0.6843
                        0.8184    0.9207    0.6971
                        0.8250    0.9237    0.7100
                        0.8317    0.9267    0.7228
                        0.8383    0.9297    0.7356
                        0.8449    0.9327    0.7484
                        0.8516    0.9357    0.7612
                        0.8582    0.9388    0.7741
                        0.8649    0.9418    0.7869
                        0.8715    0.9448    0.7997
                        0.8781    0.9478    0.8125
                        0.8848    0.9508    0.8253
                        0.8914    0.9538    0.8382
                        0.8980    0.9569    0.8510
                        0.9010    0.9573    0.8557
                        0.9040    0.9578    0.8604
                        0.9070    0.9583    0.8651
                        0.9100    0.9587    0.8698
                        0.9129    0.9592    0.8745
                        0.9159    0.9597    0.8792
                        0.9189    0.9602    0.8839
                        0.9219    0.9606    0.8886
                        0.9249    0.9611    0.8933
                        0.9278    0.9616    0.8980
                        0.9308    0.9620    0.9027
                        0.9338    0.9625    0.9075
                        0.9368    0.9630    0.9122
                        0.9398    0.9635    0.9169
                        0.9427    0.9639    0.9216
                        0.9457    0.9644    0.9263
                        0.9487    0.9649    0.9310
                        0.9517    0.9653    0.9357
                        0.9547    0.9658    0.9404
                        0.9576    0.9663    0.9451
                        0.9606    0.9667    0.9498
                        0.9636    0.9672    0.9545
                        0.9666    0.9677    0.9592
                        0.9696    0.9682    0.9639
                        0.9725    0.9686    0.9686
                        0.9736    0.9649    0.9674
                        0.9747    0.9611    0.9662
                        0.9757    0.9573    0.9650
                        0.9768    0.9535    0.9638
                        0.9778    0.9498    0.9626
                        0.9789    0.9460    0.9614
                        0.9799    0.9422    0.9602
                        0.9810    0.9385    0.9590
                        0.9821    0.9347    0.9578
                        0.9831    0.9309    0.9566
                        0.9842    0.9271    0.9554
                        0.9852    0.9234    0.9541
                        0.9863    0.9196    0.9529
                        0.9873    0.9158    0.9517
                        0.9884    0.9121    0.9505
                        0.9894    0.9083    0.9493
                        0.9905    0.9045    0.9481
                        0.9916    0.9008    0.9469
                        0.9926    0.8970    0.9457
                        0.9937    0.8932    0.9445
                        0.9947    0.8894    0.9433
                        0.9958    0.8857    0.9421
                        0.9968    0.8819    0.9409
                        0.9979    0.8781    0.9397
                        0.9989    0.8744    0.9385
                        1.0000    0.8706    0.9373
                        0.9976    0.8645    0.9340
                        0.9953    0.8584    0.9307
                        0.9929    0.8522    0.9274
                        0.9906    0.8461    0.9241
                        0.9882    0.8400    0.9208
                        0.9859    0.8339    0.9175
                        0.9835    0.8278    0.9142
                        0.9812    0.8216    0.9109
                        0.9788    0.8155    0.9076
                        0.9765    0.8094    0.9043
                        0.9741    0.8033    0.9010
                        0.9718    0.7972    0.8977
                        0.9694    0.7911    0.8944
                        0.9671    0.7849    0.8911
                        0.9647    0.7788    0.8878
                        0.9624    0.7727    0.8845
                        0.9600    0.7666    0.8813
                        0.9576    0.7605    0.8780
                        0.9553    0.7544    0.8747
                        0.9529    0.7482    0.8714
                        0.9506    0.7421    0.8681
                        0.9482    0.7360    0.8648
                        0.9459    0.7299    0.8615
                        0.9435    0.7238    0.8582
                        0.9412    0.7176    0.8549
                        0.9382    0.7077    0.8478
                        0.9351    0.6977    0.8407
                        0.9321    0.6878    0.8336
                        0.9291    0.6778    0.8265
                        0.9261    0.6679    0.8195
                        0.9231    0.6579    0.8124
                        0.9201    0.6480    0.8053
                        0.9170    0.6380    0.7982
                        0.9140    0.6281    0.7911
                        0.9110    0.6181    0.7840
                        0.9080    0.6081    0.7769
                        0.9050    0.5982    0.7698
                        0.9020    0.5882    0.7627
                        0.8989    0.5783    0.7557
                        0.8959    0.5683    0.7486
                        0.8929    0.5584    0.7415
                        0.8899    0.5484    0.7344
                        0.8869    0.5385    0.7273
                        0.8839    0.5285    0.7202
                        0.8808    0.5186    0.7131
                        0.8778    0.5086    0.7060
                        0.8748    0.4986    0.6989
                        0.8718    0.4887    0.6919
                        0.8688    0.4787    0.6848
                        0.8658    0.4688    0.6777
                        0.8627    0.4588    0.6706
                        0.8591    0.4447    0.6632
                        0.8555    0.4306    0.6558
                        0.8519    0.4165    0.6485
                        0.8483    0.4024    0.6411
                        0.8447    0.3882    0.6337
                        0.8411    0.3741    0.6264
                        0.8375    0.3600    0.6190
                        0.8339    0.3459    0.6116
                        0.8303    0.3318    0.6042
                        0.8267    0.3176    0.5969
                        0.8231    0.3035    0.5895
                        0.8195    0.2894    0.5821
                        0.8158    0.2753    0.5747
                        0.8122    0.2612    0.5674
                        0.8086    0.2471    0.5600
                        0.8050    0.2329    0.5526
                        0.8014    0.2188    0.5453
                        0.7978    0.2047    0.5379
                        0.7942    0.1906    0.5305
                        0.7906    0.1765    0.5231
                        0.7870    0.1624    0.5158
                        0.7834    0.1482    0.5084
                        0.7798    0.1341    0.5010
                        0.7762    0.1200    0.4936
                        0.7725    0.1059    0.4863
                        0.7630    0.1024    0.4802
                        0.7535    0.0989    0.4742
                        0.7440    0.0955    0.4682
                        0.7345    0.0920    0.4621
                        0.7250    0.0885    0.4561
                        0.7155    0.0851    0.4501
                        0.7060    0.0816    0.4440
                        0.6965    0.0781    0.4380
                        0.6870    0.0747    0.4320
                        0.6775    0.0712    0.4259
                        0.6680    0.0677    0.4199
                        0.6585    0.0643    0.4139
                        0.6490    0.0608    0.4078
                        0.6395    0.0573    0.4018
                        0.6300    0.0538    0.3958
                        0.6205    0.0504    0.3897
                        0.6110    0.0469    0.3837
                        0.6015    0.0434    0.3777
                        0.5920    0.0400    0.3716
                        0.5825    0.0365    0.3656
                        0.5730    0.0330    0.3596
                        0.5635    0.0296    0.3535
                        0.5540    0.0261    0.3475
                        0.5445    0.0226    0.3415
                        0.5350    0.0192    0.3354
                        0.5255    0.0157    0.3294];
        end
        end
        
        function [Vx,Vy] = computeMeanFlow(x,y,ux,uy,Xtw,Ytw,Rtw,V)
        r2 = (x-Xtw).^2 + (y-Ytw).^2;
        r2(r2 < (Rtw^2)) = 1;
        r2(r2 >= (Rtw^2)) = 0;
        Utwx = (ux.*r2)./V;
        Utwy = (uy.*r2)./V;
        Vx = mean(nonzeros(Utwx));
        Vy = mean(nonzeros(Utwy));
        end
        
        function [xT,yT] = returnLArcPos(t)
            % R = 12, Wi = 2
            x0 = -16;
            y0 = -16;
            vtw = 0.05;
            Trun = 640;
            Tstop = 120;
            Tstart = Trun + Tstop;
            TstopFinal = Tstart + Trun;
            if t < Trun
                xT = (vtw*t);
                yT = 0;
            elseif (t >= Trun) && (t < Tstart)
                xT = (vtw*Trun);
                yT = 0;
            elseif (t >= Tstart) && (t < TstopFinal)
                xT = (vtw*Trun);
                yT = (vtw*(t-Tstart));
            elseif t >= TstopFinal
                xT = (vtw*Trun);
                yT = (vtw*(TstopFinal-Tstart));
            end
            xT = x0 + xT;
            yT = y0 + yT;
        end

        function [xT,yT] = returnVArcPos(t)
            % R = 12, Wi = 2
            x0 = -20;
            y0 = -10;
            vtwx = 0.025;
            vtwy = 0.025;
            Trun = 800;
            Tstop = 100;
            Tstart = Trun + Tstop;
            TstopFinal = Tstart + Trun;
            if t < Trun
                xT = (vtwx*t);
                yT = (vtwy*t);
            elseif (t >= Trun) && (t < Tstart)
                xT = (vtwx*Trun);
                yT = (vtwy*Trun);
            elseif (t >= Tstart) && (t < TstopFinal)
                xT = (vtwx*Trun) + (vtwx*(t-Tstart));
                yT = (vtwy*Trun) - (vtwy*(t-Tstart));
            elseif t >= TstopFinal
                xT = (vtwx*Trun) + (vtwx*(TstopFinal-Tstart));
                yT = (vtwy*Trun) - (vtwy*(TstopFinal-Tstart));
            end
            xT = x0 + xT;
            yT = y0 + yT;
        end
        
        function alpha = returnSimplifiedBraidActivity(x,y,t)
        Nx = 128;
        v_tweezer = 0.035;
        yOffset = 18;
        T1 = 350;
        T2 = T1 + (2*yOffset)/v_tweezer; % 
        T3 = T2 + (1.75*yOffset)/v_tweezer; % 
        T4 = T3 + (1.75*yOffset)/v_tweezer; % 
        TstopAll = T4 + (38+12)/v_tweezer;
        
        TstartPlus = 250;
        TstopPlus = TstartPlus + 35/v_tweezer;
        
        
        % decayFactor = heaviside(-(t-325)) + heaviside(t-325)*exp(-(t-325) / 10);
        if t < 325
            decayFactor = 1;
        else
            decayFactor = exp(-(t-325) / 10);
        end
        if t < T1
            xtemp = (0.075*t);
            ytemp = (0*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            if t < TstartPlus
                alpha = -0.95*decayFactor*ones(size(x));
                alpha = alpha  -1.0*decayFactor.*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -1.0*decayFactor.*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            else
                alpha = -2.0*decayFactor.*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -2.0*decayFactor.*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            end
        elseif (t >= T1) && (t < T2)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T1)) - 13;
            ytemp = (v_tweezer*(t-T1)) - yOffset -2;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            % B = 0; %cos(7*pi/12);
            % A = -(1 - B^2)^0.5;
            alpha = -2.*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif (t >= T2) && (t < T3)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (v_tweezer*(t-T2)) - 14;
            ytemp = (0*(t-T2)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            % B = 0;
            % A = -(1 - B^2)^0.5;
            alpha = -2.*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif (t >= T3) && (t < T4)     % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T3)) + 18;
            ytemp = (-v_tweezer*(t-T3)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            % B = 0;
            % A = (1 - B^2)^0.5;
            alpha = -2.*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif t >= T4    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = -(v_tweezer*(t-T4)) + 18;
            ytemp = (0*(t-T4)) - 18;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            % B = 0;
            % A = (1 - B^2)^0.5;
            alpha = -2.*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        end
        
        
        vxTw = v_tweezer*2/sqrt(5);
        vyTw = v_tweezer*1/sqrt(5);
        if (t >= TstartPlus) && (t < TstopPlus)
            xtemp = (vxTw*(t-TstartPlus)) + 15;
            ytemp = (vyTw*(t-TstartPlus)) - yOffset - 2;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -2.0.*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.0.*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif t >= TstopPlus
            xtemp = (vxTw*(TstopPlus-TstartPlus)) + 16;
            ytemp = -11; % (vyTw*(TstopPlus-TstartPlus)) - yOffset;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -2.0.*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 6) ./ (1/4.0)) )./2.0;
            alpha = alpha -2.0.*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 6) ./ (1/4.0)) )./2.0;
        end
        
        if t >= TstopAll
            alpha = alpha.*0;
        end
        
        
        end
        %====================================================================================================================================================

        %====================================================================================================================================================
        % ACTIVITY PATTERNS
        %====================================================================================================================================================
        function alpha = activity_strip(x,Ws,Wi,a0,a1,a2)
            if nargin < 2
                a0 = -5;
                Ws = 32;
                Wi = 8;
                a1 = 0/Ws;
                a2 = 0/Ws;
            end
            alpha = a0*((1 + a1.*x + 0.5*a2.*(x.^2) ).*(tanh((x +(Ws/2.0)  )./(Wi./4.0)) - tanh((x -((Ws)/2.0)  )./(Wi/4.0)))./2);
        end

        function alpha = activity_polyWell(x,a0,wellWidth,polyDegree)
            if nargin < 2
                a0 = -5;
                wellWidth = 50;
                polyDegree = 3;
            end
            d = wellWidth / 2;
            H = 0.5*(1 + tanh( abs(x) - d ));
            normAlpha = H + (1-H).*(0.5 + 0.5.*((x./d).^polyDegree));
            alpha = a0.*normAlpha;
        end

        function alpha = activity_quadTweezerLinear(x,t,Nx,x0,y0,VX,VY,a0,a2x,a2y,R,Wi)
            if nargin < 3
                x0 = 0;
                y0 = 0;
                Nx = 128;
                a0 = -1;
                a2x = -0.02;
                a2y = 0;
                VX = 1;
                VY = 0;
                R = 5;
                Wi = 1;
            end
            xtemp = (VX*t);
            ytemp = (VY*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            alpha = a0*(1 + 0.5*( a2x*(xT-x0).^2 + a2y*(yT-y0).^2)).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
        end

        function alpha = activity_constTweezerLinear(x,t,Nx,x0,y0,VX,VY,a0,R,Wi)
            if nargin < 3
                x0 = 0;
                y0 = 0;
                Nx = 128;
                a0 = -1;
                VX = 1;
                VY = 0;
                R = 5;
                Wi = 1;
            end
            xtemp = (VX*t);
            ytemp = (VY*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            alpha = a0*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
        end
        
        function alpha = activity_oscillateInterface(x,t,a0,freq,Wi1,Wi2)
            if nargin < 3
                a0 = -5;
                freq = 0.1;
                Wi1 = 15;
                Wi2 = 35;
            end
            InterfaceWidth = Wi1 + 0.5*(1-cos(2*pi*freq*t))*(Wi2 - Wi1);
            alpha = a0*((tanh((x +((Ws)/2.0)  )./(InterfaceWidth./4.0)) - tanh((x -((Ws)/2.0)  )./(InterfaceWidth./4.0)))./2);
        end

        function alpha = activity_surf(x,t,Nx,a0,vwx,Ws,Wi)
            if nargin < 3
                Nx = 128;
                a0 = -4;
                vwx = 1;
                Ws = 20;
                Wi = 10;
            end
            alpha = a0.*(tanh((mod(x+(Nx/2)-(vwx*t),Nx) - Nx/2 +((Ws)/2.0)  )./(Wi./4.0)) - tanh((mod(x+(Nx/2)-(vwx*t),Nx) - Nx/2 -((Ws)/2.0)  )./(Wi./4.0)))./2;
        end

        function alpha = activity_ratchet(x,t)
            a0 = -5;
            freq = 0.01;
            Wi1 = 10;
            Wi2 = 20;
            Wo = 0;
            Ws = 25;
            delta = 0.5*(1-cos(2*pi*freq*t))*(Wi2 - Wi1);
            alpha = a0*(tanh((x - Wo +((Ws)/2.0)  )./(Wi1./16.0)) - tanh((x - delta - Wo -((Ws)/2.0)  )./(Wi1/4.0)))./2;
        end

        function alpha = activity_rampTweezer(x,y,a0,a1x,a1y,R,Wi)
        if nargin < 3
            a0 = -2.5;
            a1x = 0;
            a1y = 0.9;
            R = 7.5;
            Wi = 1;
        end
        alpha = a0*(1 + a1x*x/R + a1y*y/R).*(1 - tanh( ( sqrt(x.^2 + y.^2) - R ) ./ (Wi/4.0)))./2.0;
        end

        function alpha = activity_annulus(x,y,R,Ws,Wi,a0)
            if nargin < 3
                R = 20;
                Ws = 4;
                Wi = 1;
                a0 = -2.5;
            end
        alpha = a0*(tanh( ( sqrt(x.^2 + y.^2) - R + (Ws/2.0)) ./ (Wi/4.0)) - tanh( ( sqrt(x.^2 + y.^2) - R - (Ws/2.0)) ./ (Wi/4.0)) )./2.0;
        end

        function alpha = activity_annulRotate(x,y,t,a0,R,Ws,Wi,phi0,freq)
            if nargin < 3
                t = 0;
                a0 = -5;
                phi0 = 0;
                freq = 0;
                R = 20;
                Ws = 2;
                Wi = 1;
            end
            alpha = a0*(0.5 + 0.5*sin(atan2(y,x)- phi0 - 2*pi*freq*t) ).*(tanh( ( sqrt(x.^2 + y.^2) - R + (Ws./2.0)) ./ (Wi/4.0)) - tanh( ( sqrt(x.^2 + y.^2) - R - (Ws./2.0)) ./ (Wi/4.0)) )./2.0;
        end
        
        function alpha = activity_parabolicL(x,y,t,Nx,vtw,x0,y0,R,Wi,a0,a1x,a2Start,Trun,Tstop)
            if nargin < 4
                Nx = 128;
                x0 = -32;
                y0 = 0;
                R = 9;
                Wi = 1;
                vtw = 0.05;
                a0 = -8;
                a1x = 0;
                a2Start = -0.02;
                Trun = 800;
                Tstop = 160;
            end
            a2 = a2Start;
            Tstop1 = Trun;
            Tstart1 = Tstop1 + Tstop;
            TstopFinal = Tstart1 + Trun;
            xtemp = (vtw*t);
            ytemp = 0*(vtw*t);
            if (t >= Tstop1) && (t < Tstart1)
                xtemp = (vtw*Tstop1);
                ytemp = 0*(vtw*Tstop1);
                a2 = a2Start*cos(pi*(t-Tstop1)/(Tstart1-Tstop1));
            elseif (t >= Tstart1) && (t < TstopFinal)
                xtemp = (vtw*Tstop1) + 0*(vtw*(t-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(t-Tstart1));
                a2 = -a2Start;
            elseif t >= TstopFinal
                xtemp = (vtw*Tstop1) + 0*(vtw*(TstopFinal-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(TstopFinal-Tstart1));
                a2 = -a2Start;
            end
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            a2x = (a2Start + a2)/2;
            a2y = (a2Start - a2)/2;
            alpha = (a0 + a1x.*(xT-x0) + 0.5*(a2x.*(xT-x0).^2 + a2y.*(xT-x0).*(yT-y0))).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            % alpha = a0*(1.0 + 0.5*(a2x.*(xT-x0).^2 + a2y.*(yT-y0).^2)).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            if t > TstopFinal
                alpha = alpha.*exp(-(t-TstopFinal)/(10));
            end
        end

        function alpha = activity_hyperbolicArc(x,y,t,Nx,VX,VY,x0,y0,R,Wi,a0,a2Start,Trun,Tstop)
        if nargin < 4
            Nx = 128;
            x0 = 48;
            y0 = 0;
            R = 9;
            Wi = 1;
            VX = -0.08;
            VY = 0.04;
            a0 = -2;
            a2Start = -0.1;
            Trun = 536;
            Tstop = 100;
        end
        a2 = a2Start;
        Tstop1 = Trun;
        Tstart1 = Tstop1 + Tstop;
        TstopFinal = Tstart1 + Trun;
        xtemp = (VX*t);
        ytemp = (VY*t);
        if (t >= Tstop1) && (t < Tstart1)
            xtemp = (VX*Tstop1);
            ytemp = (VY*Tstop1);
            a2 = a2Start*cos(pi*(t-Tstop1)/(Tstart1-Tstop1));
        elseif (t >= Tstart1) && (t < TstopFinal)
            xtemp = (VX*Tstop1) + (VX*(t-Tstart1));
            ytemp = (VY*Tstop1) - (VY*(t-Tstart1));
            a2 = -a2Start;
        elseif t >= TstopFinal
            xtemp = (VX*Tstop1) + (VX*(TstopFinal-Tstart1));
            ytemp = (VY*Tstop1) - (VY*(TstopFinal-Tstart1));
            a2 = -a2Start;
        end
        xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
        yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
        alpha = a0*(1.0 + 0.5*a2*((xT-x0).*(yT-y0))).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
        if t > TstopFinal
            alpha = alpha.*exp(-(t-TstopFinal)/(10));
        end
        end

        function alpha = activity_braid(x,y,t)
        % Braid a pair of defects
        Nx = 128;
        v_tweezer = 0.075;
        yOffset = 18;
        T1 = 350;
        T2 = T1 + (yOffset+12)/v_tweezer; % 883;
        T3 = T2 + (18+12)/v_tweezer; % 1283;
        T4 = T3 + (8+12)/v_tweezer; % 1683;
        TstopAll = T4 + (38+12)/v_tweezer;
        
        TstartPlus = 150;
        TstopPlus = TstartPlus + 44.72/v_tweezer;
        
        
        % decayFactor = heaviside(-(t-325)) + heaviside(t-325)*exp(-(t-325) / 10);
        if t < 325
            decayFactor = 1;
        else
            decayFactor = exp(-(t-325) / 10);
        end
        if t < T1
            xtemp = (v_tweezer*t);
            ytemp = (0*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            if t < TstartPlus
                alpha = -1.0*decayFactor*ones(size(x));
                alpha = alpha  -4.0*decayFactor*(1 + 2*0.25*xT/16).*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -4.0*decayFactor*(1 - 2*0.25*xTm/16).*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            else
                alpha = -4.0*decayFactor*(1.25 + 2*0.25*xT/16).*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -4.0*decayFactor*(1.25 - 2*0.25*xTm/16).*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            end
        elseif (t >= T1) && (t < T2)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T1)) - 18;
            ytemp = (v_tweezer*(t-T1)) - yOffset;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            
            alpha = -5*(1 - 2*1*((yT-y0)/10).^2 ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
            alpha = alpha -4*(1 + 2*1*((yTm+y0)/10).^2 ).*(1 - tanh( ( sqrt((xTm-x0).^2 + (yTm-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
        elseif (t >= T2) && (t < T3)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (v_tweezer*(t-T2)) - 18;
            ytemp = (0*(t-T2)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            
            alpha = -5*(1 - 2*1*((xT-x0)/10).^2 ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
            alpha = alpha -4*(1 + 2*1*((xTm+x0)/10).^2 ).*(1 - tanh( ( sqrt((xTm-x0).^2 + (yTm-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
        elseif (t >= T3) && (t < T4)     % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T3)) + 12;
            ytemp = (-v_tweezer*(t-T3)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            
            alpha = -4*(1 + 2*1*((yT-y0)/10).^2 ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
            alpha = alpha -5*(1 - 2*1*((yTm+y0)/10).^2 ).*(1 - tanh( ( sqrt((xTm-x0).^2 + (yTm-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
        elseif t >= T4    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = -(v_tweezer*(t-T4)) + 12;
            ytemp = (0*(t-T4)) - 8;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            
            alpha = -4*(1 + 2*1*((xT-x0)/10).^2 ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
            alpha = alpha -5*(1 - 2*1*((xTm+x0)/10).^2 ).*(1 - tanh( ( sqrt((xTm-x0).^2 + (yTm-y0).^2) - (10./2.0)) ./ (1/4.0)) )./2.0;
        end
        
        
        a2 = 0.075;
        vxTw = v_tweezer*2/sqrt(5);
        vyTw = v_tweezer*1/sqrt(5);
        if (t >= TstartPlus) && (t < TstopPlus)
            xtemp = (vxTw*(t-TstartPlus)) + 10;
            ytemp = (vyTw*(t-TstartPlus)) - yOffset;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -2.0*(1.0 + 0.5*a2*((xT).*(yT))).*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 5) ./ (1/4.0)) )./2.0;
            alpha = alpha -2.0*(1.0 + 0.5*a2*((xTm).*(yTm))).*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 5) ./ (1/4.0)) )./2.0;
        elseif t >= TstopPlus
            xtemp = (vxTw*(TstopPlus-TstartPlus)) + 10;
            ytemp = (vyTw*(TstopPlus-TstartPlus)) - yOffset;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -1.0*(1.0 + 0*a2*((xT).*(yT))).*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 5) ./ (1/4.0)) )./2.0;
            alpha = alpha -1.0*(1.0 + 0*a2*((xTm).*(yTm))).*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 5) ./ (1/4.0)) )./2.0;
        end
        
        if t >= TstopAll
            alpha = alpha.*exp(-(t-T4)/(10));
        end
        
        
        end
    
        function alpha = activity_hourglassTweezer(x,y,t,Nx,vtw,x0,y0,R,Wi,a0,a1x,a2Start,Trun,Tstop)
            if nargin < 4
                Nx = 128;
                x0 = -32;
                y0 = 0;
                R = 9;
                Wi = 1;
                vtw = 0.05;
                a0 = -8;
                a1x = 0;
                a2Start = -0.02;
                Trun = 800;
                Tstop = 160;
            end
            a2 = a2Start;
            Tstop1 = Trun;
            Tstart1 = Tstop1 + Tstop;
            TstopFinal = Tstart1 + Trun;
            xtemp = (vtw*t);
            ytemp = 0*(vtw*t);
            if (t >= Tstop1) && (t < Tstart1)
                xtemp = (vtw*Tstop1);
                ytemp = 0*(vtw*Tstop1);
                a2 = a2Start*cos(pi*(t-Tstop1)/(Tstart1-Tstop1));
            elseif (t >= Tstart1) && (t < TstopFinal)
                xtemp = (vtw*Tstop1) + 0*(vtw*(t-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(t-Tstart1));
                a2 = -a2Start;
            elseif t >= TstopFinal
                xtemp = (vtw*Tstop1) + 0*(vtw*(TstopFinal-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(TstopFinal-Tstart1));
                a2 = -a2Start;
            end
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            a2x = (a2Start + a2)/2;
            a2y = (a2Start - a2)/2;
            alpha = -0.1 + 0.5*(1 + tanh((abs(xT-x0) - abs(yT-y0))/(Wi/4)) ).*(a0 + a1x.*(xT-x0) + 0.5*(a2x.*(xT-x0).^2 + a2y.*(xT-x0).*(yT-y0))).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            % alpha = a0*(1.0 + 0.5*(a2x.*(xT-x0).^2 + a2y.*(yT-y0).^2)).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            if t > TstopFinal
                alpha = alpha.*exp(-(t-TstopFinal)/(10));
            end
        end
    
        function alpha = activity_angularHarmonic(x,y,t,Nx,vtw,x0,y0,R,Wi,a0,angA,angB,Trun,Tstop)
            if nargin < 4
                Nx = 128;
                x0 = -32;
                y0 = 0;
                R = 9;
                Wi = 1;
                vtw = 0.05;
                a0 = -8;
                a1x = 0;
                a1y = 0;
                a2Start = -0.02;
                Trun = 800;
                Tstop = 160;
            end
            a2Start = 0;
            a2 = a2Start;
            Tstop1 = Trun;
            Tstart1 = Tstop1 + Tstop;
            TstopFinal = Tstart1 + Trun;
            xtemp = (vtw*t);
            ytemp = 0*(vtw*t);
            if (t >= Tstop1) && (t < Tstart1)
                xtemp = (vtw*Tstop1);
                ytemp = 0*(vtw*Tstop1);
                a2 = a2Start*cos(pi*(t-Tstop1)/(Tstart1-Tstop1));
            elseif (t >= Tstart1) && (t < TstopFinal)
                xtemp = (vtw*Tstop1) + 0*(vtw*(t-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(t-Tstart1));
                a2 = -a2Start;
            elseif t >= TstopFinal
                xtemp = (vtw*Tstop1) + 0*(vtw*(TstopFinal-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(TstopFinal-Tstart1));
                a2 = -a2Start;
            end
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            a2x = (a2Start + a2)/2;
            a2y = (a2Start - a2)/2;
            alpha = a0.*(1 + angA*sin(2.*atan2(yT-y0,xT-x0)) + angB*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            if t > TstopFinal
                alpha = alpha.*exp(-(t-TstopFinal)/(10));
            end
        end

        function alpha = activity_angHarmLarc(x,y,t,Nx,vtw,x0,y0,R,Wi,a0,Trun,Tstop)
            if nargin < 4
                Wi = 2;
                x0 = -12;
                y0 = -12;
                R = 12;
                vtw = 0.05;
                a0 = -5;
                Trun = 480;
                Tstop = 120;
            end
            amp = a0;
            Tstop1 = Trun;
            Tstart1 = Tstop1 + Tstop;
            TstopFinal = Tstart1 + Trun;
            if t < Tstop1
                xtemp = (vtw*t);
                ytemp = 0*(vtw*t);
                B = 1;
            elseif (t >= Tstop1) && (t < Tstart1)
                xtemp = (vtw*Tstop1);
                ytemp = 0*(vtw*Tstop1);
                if (t-Tstop1)/(Tstart1-Tstop1) < 0.33
                    B = 1;
                    amp = a0*(cos(pi*(t-Tstop1)/(0.67*(Tstart1-Tstop1))))^2;
                elseif 0.33 < (t-Tstop1)/(Tstart1-Tstop1) && (t-Tstop1)/(Tstart1-Tstop1) < 0.67
                    B = cos(2*pi/3);
                    amp = a0*(cos(pi*(t-Tstop1)/(0.67*(Tstart1-Tstop1))))^2;
                else
                    B = cos(2*pi/3);
                    amp = a0;
                end
            elseif (t >= Tstart1) && (t < TstopFinal)
                xtemp = (vtw*Tstop1) + 0*(vtw*(t-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(t-Tstart1));
                B = cos(2*pi/3);
            elseif t >= TstopFinal
                xtemp = (vtw*Tstop1) + 0*(vtw*(TstopFinal-Tstart1));
                ytemp = 0*(vtw*Tstop1) + (vtw*(TstopFinal-Tstart1));
                B = cos(2*pi/3);
            end
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            A = -(1 - B^2)^0.5;
            alpha = amp.*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            if t > TstopFinal
                alpha = alpha.*exp(-(t-TstopFinal)/(10));
            end
        end

        function alpha = activity_angHarmVarc(x,y,t,Nx,vtwx,vtwy,x0,y0,R,Wi,a0,Trun,Tstop)
            if nargin < 4
                Wi = 2;
                x0 = -20;
                y0 = -10;
                R = 12;
                vtwx = 0.025;
                vywY = 0.025;
                a0 = -0.5;
                Trun = 800;
                Tstop = 100;
            end
            amp = a0; 
            Tstop1 = Trun;
            Tstart1 = Tstop1 + Tstop;
            TstopFinal = Tstart1 + Trun;
            if t < Tstop1
                xtemp = (vtwx*t);
                ytemp = (vtwy*t);
                A = 1;
            elseif (t >= Tstop1) && (t < Tstart1)
                xtemp = (vtwx*Tstop1);
                ytemp = (vtwy*Tstop1);
                if (t-Tstop1)/(Tstart1-Tstop1) < 0.5
                    A = 1;
                    amp = a0*(cos(pi*(t-Tstop1)/(1*(Tstart1-Tstop1))))^2;
                else
                    A = -1;
                    amp = a0*(cos(pi*(t-Tstop1)/(1*(Tstart1-Tstop1))))^2;
                end
            elseif (t >= Tstart1) && (t < TstopFinal)
                xtemp = (vtwx*Tstop1) + (vtwx*(t-Tstart1));
                ytemp = (vtwy*Tstop1) - (vtwy*(t-Tstart1));
                A = -1;
            elseif t >= TstopFinal
                xtemp = (vtwx*Tstop1) + (vtwx*(TstopFinal-Tstart1));
                ytemp = (vtwy*Tstop1) - (vtwy*(TstopFinal-Tstart1));
                A = -1;
            end
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            B = 0;
            alpha = amp.*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;
            if t > TstopFinal
                alpha = alpha.*exp(-(t-TstopFinal)/(10));
            end

        end
    
        function alpha = activity_angHarmLinear(x,y,t,Nx,x0,y0,VX,VY,a0,angA,angB,R,Wi)
            if nargin < 3
                x0 = 0;
                y0 = 0;
                Nx = 128;
                a0 = -1;
                angA = 0;
                angB = 1;
                VX = 1;
                VY = 0;
                R = 5;
                Wi = 1;
            end
            xtemp = (VX*t);
            ytemp = (VY*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            alpha = a0.*(1 + angA*sin(2.*atan2(yT-y0,xT-x0)) + angB*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;        
            % alpha = a0.*(1-exp(-sqrt((xT-x0).^2 + (yT-y0).^2)/(0.25*Wi))).*(1 + angA*sin(2.*atan2(yT-y0,xT-x0)) + angB*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - R) ./ (Wi/4.0)) )./2.0;        
        end

        function alpha = activity_angHarmBraid(x,y,t)
        Nx = 128;
        v_tweezer = 0.035;
        yOffset = 18;
        T1 = 350;
        T2 = T1 + (2*yOffset)/v_tweezer; % 
        T3 = T2 + (1.75*yOffset)/v_tweezer; % 
        T4 = T3 + (1.75*yOffset)/v_tweezer; % 
        TstopAll = T4 + (38+12)/v_tweezer;
        
        TstartPlus = 250;
        TstopPlus = TstartPlus + 35/v_tweezer;
        
        
        % decayFactor = heaviside(-(t-325)) + heaviside(t-325)*exp(-(t-325) / 10);
        if t < 325
            decayFactor = 1;
        else
            decayFactor = exp(-(t-325) / 10);
        end
        if t < T1
            xtemp = (0.075*t);
            ytemp = (0*t);
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            if t < TstartPlus
                alpha = -2.0*decayFactor*ones(size(x));
                alpha = alpha  -10.0*decayFactor*(1 + 2*0.25*xT/16).*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -10.0*decayFactor*(1 - 2*0.25*xTm/16).*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            else
                alpha = -12.0*decayFactor*(1 + 2*0.25*xT/16).*(1 - tanh( ( sqrt(0.25*xT.^2 + (yT-yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
                alpha = alpha  -12.0*decayFactor*(1 - 2*0.25*xTm/16).*(1 - tanh( ( sqrt(0.25*xTm.^2 + (yTm+yOffset).^2) - (16./4.0)) ./ (1/4.0)) )./2.0;
            end
        elseif (t >= T1) && (t < T2)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T1)) - 13;
            ytemp = (v_tweezer*(t-T1)) - yOffset -2;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            B = 0; %cos(7*pi/12);
            A = -(1 - B^2)^0.5;
            alpha = -5.5*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -5.5*(1 + A*sin(2.*atan2(yTm+y0,xTm+x0)) + B*cos(2.*atan2(yTm+y0,xTm+x0)) ).*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif (t >= T2) && (t < T3)    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (v_tweezer*(t-T2)) - 14;
            ytemp = (0*(t-T2)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            B = 0;
            A = -(1 - B^2)^0.5;
            alpha = -5.5*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -5.5*(1 + A*sin(2.*atan2(yTm+y0,xTm+x0)) + B*cos(2.*atan2(yTm+y0,xTm+x0)) ).*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif (t >= T3) && (t < T4)     % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = (0*(t-T3)) + 18;
            ytemp = (-v_tweezer*(t-T3)) + 12;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            B = 0;
            A = (1 - B^2)^0.5;
            alpha = -5.5*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -5.5*(1 + A*sin(2.*atan2(yTm+y0,xTm+x0)) + B*cos(2.*atan2(yTm+y0,xTm+x0)) ).*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif t >= T4    % +1/2 defect at (-40,20), -1/2 defect at (15,32)
            xtemp = -(v_tweezer*(t-T4)) + 18;
            ytemp = (0*(t-T4)) - 18;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            x0 = 0;
            y0 = 0;
            B = 0;
            A = (1 - B^2)^0.5;
            alpha = -5.5*(1 + A*sin(2.*atan2(yT-y0,xT-x0)) + B*cos(2.*atan2(yT-y0,xT-x0)) ).*(1 - tanh( ( sqrt((xT-x0).^2 + (yT-y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -5.5*(1 + A*sin(2.*atan2(yTm+y0,xTm+x0)) + B*cos(2.*atan2(yTm+y0,xTm+x0)) ).*(1 - tanh( ( sqrt((xTm+x0).^2 + (yTm+y0).^2) - 9.0) ./ (2/4.0)) )./2.0;
        end
        
        
        vxTw = v_tweezer*2/sqrt(5);
        vyTw = v_tweezer*1/sqrt(5);
        if (t >= TstartPlus) && (t < TstopPlus)
            xtemp = (vxTw*(t-TstartPlus)) + 15;
            ytemp = (vyTw*(t-TstartPlus)) - yOffset - 2;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -2.0*(1 + sin(2.*atan2(yT,xT)) ).*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 9.0) ./ (2/4.0)) )./2.0;
            alpha = alpha -2.0*(1 + sin(2.*atan2(yTm,xTm)) ).*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 9.0) ./ (2/4.0)) )./2.0;
        elseif t >= TstopPlus
            xtemp = (vxTw*(TstopPlus-TstartPlus)) + 16;
            ytemp = -11; % (vyTw*(TstopPlus-TstartPlus)) - yOffset;
            xT = mod(x-xtemp+(Nx/2),Nx) - Nx/2;
            yT = mod(y-ytemp+(Nx/2),Nx) - Nx/2;
            xTm = mod(x+xtemp+(Nx/2),Nx) - Nx/2;
            yTm = mod(y+ytemp+(Nx/2),Nx) - Nx/2;
            alpha = alpha -1.0*(1 + sin(2.*atan2(yT,xT)) ).*(1 - tanh( ( sqrt((xT).^2 + (yT).^2) - 6) ./ (1/4.0)) )./2.0;
            alpha = alpha -1.0*(1 + sin(2.*atan2(yTm,xTm)) ).*(1 - tanh( ( sqrt((xTm).^2 + (yTm).^2) - 6) ./ (1/4.0)) )./2.0;
        end
        
        if t >= TstopAll
            alpha = alpha.*0;
        end
        
        
        end
    end
end

