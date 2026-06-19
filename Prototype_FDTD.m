% 2D FDTD Simulation
% Ziyao Su

%% STEP 1: Basic FDTD and TFSF Implementation
%{ 
# Defined the Laser and Grid class.
# Defined seperate plot function that can verify the temporal and
spatial profiles of the source.
# Combined p-pol and s-pol FDTD loops in one code and can be simply switched.
%}

% The axis defined in this code is: (→ y) and (↑ x), thus (× z).
% The source incident angle in this work can only range from 0 to 90 degrees.


%% STEP 2: PML Implementation
%{
# Defined the PML class using unsplit method with polynomial grading.
# Defined a function that can check the PML implementation.
# Defined a second plot function that can verify the reflected pulse intensity
profile of the source.
# Defined a function that calculate the total energy vs. theoretical 
pulse energy and output the result in txt file.
%}


%% STEP 3: Material Implementation
%{
# Defined the Material class including dielectric and metal.
# Defined the Geometry class including structure of multilayer mirror and grating.
# Expanded the basic FDTD with Drude model.
# Defined three more plot functions that can verify the total energy
conservation for metals and dielectrics and plot multilayer mirror
contours.
# Defined a function to name the stored movie.
%}

% !special thanks to Dr.Niezgoda and Dr.Rumpf for helping me for implementation on metal 
% target in this section!


%% STEP 4: Physics Model Implementation
%{
# Defined the two physics model functions including 2 different impact ionization models, 
and 1 Lorentz collision model.
# Added Keldysh photoionization model, Ohmic heating model, and Low-freq Lorentz model in the main
loop, with effective photoionization current updated.
# Moved the Drude current term outside the FDTD loop to update separately compared to STEP 3.
# Defined a plot function that plot electron density, E field, different rates.
%}

% !This section is mainly inspired by Dr. Simin's FDTD work!
% ver1 keeps the half step feature in main loop as what Dr. Simin's work did, while ver2 updates all
% parameters once in each loop


%% STEP 5: Polishment and Improvement
%{
# Added MRE level parameter and more electronic properties in Material class.
# Defined MRE class.
# Defined two Drude current functions, and a metal electron diffusion function. 
%}


% References
% [1] Allen Taflove, Susan C. Hagness - Computational Electrodynamics The Finite-Difference Time-Domain Method (2000)
% [2] Benavides-Cruz. Numerical simulation of metallic nanostructures interacting with electromagnetic fields using the Lorentz–Drude model and FDTD method (2015) 
% [3] Refrective index information. https://refractiveindex.info/
% [4] To, N., Juodkazis, S., & Nishijima, Y. Detailed Experiment-Theory Comparison of Mid-Infrared Metasurface Perfect Absorbers. Micromachines, 11(4), 409 (2020)
% [5] Fresnel equations. https://en.wikipedia.org/wiki/Fresnel_equations
% [6] James D. Callen - Fundamentals of Plasma Physics - Draft(2006)
% [7] Peñano, J.P.; Sprangle, P.; Haﬁzi, B.; Manheimer, W.; Zigler, A. Transmission of intense femtosecond laser pulses into dielectrics. Phys. Rev. E 72, 036412 (2005).
% [8] P.-J. CHARPIN. Simulation of laser-induced ionization in wide bandgap solid dielectrics with a particle-in-cell code. Optics Express Vol. 32, No. 6, 10175 (2024).
% [9] L. V. Keldysh, "Ionization in the field of a strong electromagnetic wave," J. Exp. Theor.Phys., vol. 20, no. 5, pp. 1307–1314, 1965
% [10] L. V. Keldysh, "Kinetic theory of impact ionization in semiconductors," SOVIET PHYSICS JETP, vol. 37, no. 10, 1960.
% [11] Exarhos, G. J. Laser-Induced Damage in Optical Materials: 2003. Proceedings of SPIE Vol. 5273. SPIE, Bellingham, WA. (2004)
% [12] N. Sano and A. Yoshii, "Impact-ionization theory consistent with a realistic band structure of silicon,” Phys. Rev. B 45, 4171–4180 (1992)

%% -------------------------------------------------------------------
clear; close all
cN=40;
maxNumCompThreads(cN)


%% Setup paths
projectRoot = '/fs/ess/PAS1730/FDTD';

addpath(genpath(fullfile(projectRoot, 'Functions')));
addpath(genpath(fullfile(projectRoot, 'Classes')));
addpath(genpath(fullfile(projectRoot, 'Inputs')));


%% physical parameters
e       = 1.602*10^-19;     % [C] electron charge
me      = 9.109*10^-31;     % [kg] electron mass
NA      = 6.0221409e+23;    % [#] Avogadro's number
kb      = 1.38064852*10^-23;% [J/K] 
c       = 299.792458;       % [nm/fs] speed of light
ep0     = 8.854187817e-9;   % [C^2*fs^2/(kg*nm^3)] vacuum permittivity
mu0     = 1/c^2/ep0;        % vacuum permeability


%% Create a Grid class (nx[#], ny[#], d[nm], T[fs])
myGrid = Grid(2000, 1100, 12, 170);

nx   = myGrid.nx;
ny   = myGrid.ny;
d    = myGrid.d;                % [nm] grid spacing in nm along x, y
dx   = d;
dy   = d;

dt   = myGrid.dt;               % [fs] time step
T    = myGrid.T;                % [fs] total time
Tsteps  = myGrid.Tsteps;        % [#] total number of time steps;


%% Create a Laser class (wavelength[nm], pulseDuration[fs], spotRadius[nm], fluence[J/cm^2], incidenctAngle[degree], polarization[p or s])
% spotRadius = 12*wavelength when wavelength = 800nm could ensure that the peak loss within ~200fs is less than 2%.  
myLaser = Laser(1950, 70, 4500, 0.45, <aoi>, 's');
aoi = myLaser.incidenctAngle;
pol = myLaser.polarization;
omega = myLaser.omega;
fluence = myLaser.fluence;


%% Initialize TFSF Sources
% Initialize positions
tfsf_y = round(ny/4*0.9);           % [#] TFSF y position
pulseCenter = round(nx/2);      % [#] pulse transverse peak center position
pulseWidth = 2.0*myLaser.spotRadius/dx;  % [#] approximation of pulse transverse width radius for normal incidence (don't need to round here)
pulseWidth_aoi = round(pulseWidth/cosd(aoi));  % [#] approximation of pulse transverse width radius for theta incidence on TFSF boundary. Divide cos not times cos!

% Calculate phase velocity and delay time
phi_theta = myLaser.phaseV(myGrid);   % [nm/fs] adjusted phase velocity along propogation direction
tau_delay = round(pulseWidth*d*tand(aoi)/phi_theta/dt);  % [#] delay time steps from the transverse edge to the peak center of the pulse 
delay = myLaser.pulseDuration + tau_delay*dt*2; % [fs] usually setting 2~3 times of tau_delay 

% Calculate source profile on TFSF boundary
x = 1:nx;
TFSF_profile1 = TFSF_functions.TFSF_bufferWindow(x, pulseCenter, pulseWidth_aoi); 
TFSF_profile2 = TFSF_functions.TFSF_Gaussian(x, myGrid, myLaser, pulseCenter);

% Initialize time steps and Sources in terms of t
source_t = 0:dt:2*T;
% Et = myLaser.GaussianEt(source_t, delay).*myLaser.CarrierEt(source_t);
% Ht = myLaser.GaussianHt(dt,source_t, delay).*myLaser.CarrierHt(dt, source_t);
Et = myLaser.CosineEt(source_t, delay).*myLaser.CarrierEt(source_t);
Ht = myLaser.CosineHt(dt, source_t, delay).*myLaser.CarrierHt(dt, source_t);

% Store Sources at TFSF in 2D array for each time step
if strcmp(pol, 's')
    % s-pol
    Ez_TFSF = TFSF_functions.TFSF_injectionAmplitude(myGrid, myLaser, phi_theta, pulseCenter, pulseWidth_aoi, tau_delay, Et, TFSF_profile1, TFSF_profile2);
    Hx_TFSF = TFSF_functions.TFSF_injectionAmplitude(myGrid, myLaser, phi_theta, pulseCenter, pulseWidth_aoi, tau_delay, Ht, TFSF_profile1, TFSF_profile2)...
        *cosd(aoi);
    Ex_TFSF = zeros(nx, Tsteps);
    Hz_TFSF = zeros(nx, Tsteps);
else
    % p-pol
    Ez_TFSF = zeros(nx, Tsteps);
    Hx_TFSF = zeros(nx, Tsteps);
    Ex_TFSF = -TFSF_functions.TFSF_injectionAmplitude(myGrid, myLaser, phi_theta, pulseCenter, pulseWidth_aoi, tau_delay, Et, TFSF_profile1, TFSF_profile2)...
         *cosd(aoi);
    Hz_TFSF = TFSF_functions.TFSF_injectionAmplitude(myGrid, myLaser, phi_theta, pulseCenter, pulseWidth_aoi, tau_delay, Ht, TFSF_profile1, TFSF_profile2);
end


%% Create Materials class (type[d or m], n[#], kappa[#], mur[#], rou[g/cm^3], mw[g/mol], Z[#], bg[eV], me[#], mh[#], omega[rad/fs])
% n and kappa info can be found in Ref[3], just use
% the n and kappa that ensures ep_infinity > 0 for metals, though
% ep_infinity has < 1% influences in Drude model[4]
% At 1950nm wavelength:
mySiO2 = Material('name', 'SiO2', 'type', 'd', 'n', 1.45, 'rou', 2.202, 'mw', 60.08, 'Z', 4, 'bg', 9, 'me', 0.6, 'mh', 8, 'omega', omega);  % with optional inputs
myHfO2 = Material('name', 'HfO2', 'type', 'd', 'n', 1.922, 'rou', 9.68, 'mw', 210.49, 'Z', 4, 'bg', 5.7, 'me', 1.09, 'mh', 1.12, 'omega', omega);  % with optional inputs

% load in photoionization data:
mySiO2 = mySiO2.loadKeldyshData('Inputs/RatesNFields_SiO2_1950nm.mat', 'Inputs/EffBgNFields_SiO2_1950nm.mat');
myHfO2 = myHfO2.loadKeldyshData('Inputs/RatesNFields_HfO2_1950nm.mat', 'Inputs/EffBgNFields_HfO2_1950nm.mat');
mySiO2.trec = 220;
myHfO2.trec = 1050;

myMaterial = [mySiO2, myHfO2];
[myMaterialNames, numMetals] = SavingData_functions.assignMaterialNames(myMaterial);

% figure();  
% subplot(2,1,1)
% plot(mySiO2.log10E_Keldysh,mySiO2.log10wpho_Keldysh)
% title('logRates of SiO2')
% subplot(2,1,2)
% plot(mySiO2.E_Keldysh,mySiO2.bgeff_Keldysh/e)
% title('Bandgaps (eV) of SiO2')


%% Create a PML class (PML thickness[#], PML coeffecient[#], expected reflection order[#], Grid object[class])
% Usually set m = [3,4], and 10 to 20 cells for the PML layer. The
% residual intensity is always around 1e-8. Each reflection from the PML
% can reduce around 1e-4 of the intensity in this code.
% Ref[1] p306
myPML = PML(20, 3, 1e-8, myGrid);
PMLthick_cell = myPML.PMLn;

% common coefficients for s-pol and p-pol
k_BDz_1 = myPML.k_BDz_1; k_BDz_2 = myPML.k_BDz_2;
k_BDx_1 = myPML.k_BDx_1; k_BDx_2 = myPML.k_BDx_2; 
k_BDy_1 = myPML.k_BDy_1; k_BDy_2 = myPML.k_BDy_2;

% s-pol
k_Ez_1 = myPML.k_Ez_1; k_Ez_2 = myPML.k_Ez_2; 
k_Hx_1 = myPML.k_Hx_1; k_Hx_2 = myPML.k_Hx_2; 
k_Hy_0 = myPML.k_Hy_0; k_Hy_1 = myPML.k_Hy_1; k_Hy_2 = myPML.k_Hy_2; 

% p-pol
k_Hz_1 = myPML.k_Hz_1; k_Hz_2 = myPML.k_Hz_2; 
k_Ex_1 = myPML.k_Ex_1; k_Ex_2 = myPML.k_Ex_2; 
k_Ey_0 = myPML.k_Ey_0; k_Ey_1 = myPML.k_Ey_1; k_Ey_2 = myPML.k_Ey_2;

% Check coefficients
% Analysis_functions.plot_PML_Coef(myPML);


%% Initialize Global FDTD Fields
% s-pol
Ez = zeros(nx,ny);
Dz = zeros(nx,ny);
Jdz = zeros(nx,ny);
Jpz = zeros(nx,ny);
Pz = zeros(nx,ny);
Bx = zeros(nx,ny);
Hx = zeros(nx,ny);
By = zeros(nx,ny);
Hy = zeros(nx,ny);
% p-pol
Bz = zeros(nx,ny);
Hz = zeros(nx,ny);
Ex = zeros(nx,ny);
Dx = zeros(nx,ny);
Jdx = zeros(nx,ny);
Jpx = zeros(nx,ny);
Px = zeros(nx,ny);
Ey = zeros(nx,ny);
Dy = zeros(nx,ny);
Jdy = zeros(nx,ny);
Jpy = zeros(nx,ny);
Py = zeros(nx,ny);


%% Create Geometry class (Grid object[class])
myTarget = Geometry(myGrid);
 
% === 1. Define layer stack === 
% ---bulk target---
% material_ids = [1];
% material_thick  = [1000];  % [nm]

% ---multilayer target---
% material_ids = load('Inputs/types.txt');        
material_thick  = load('Inputs/thicknesses2.txt');  % [nm]
material_thick  = flipud(material_thick)';

material_ids = repmat([1 2], 1, 13); 
% material_thick  = repmat([138 101], 1, 10);  % [nm]
% % add other layers
material_ids = [material_ids, 1];         % material ID

material_thick_cell = round(material_thick / dy);  % [#]
% ---grating target---
duty_cycle = 1-<dc>;           % [#] duty cycle of the groove at the bottom
lines_per_mm = 630;        % [#/mm] number of pillar lines per mm
struc_period = 1e6 / lines_per_mm;            % [nm] structure period
groove_width  = duty_cycle * struc_period;    % [nm] groove width
pillar_depth = <tp>;           % [nm] pillar depth of the grating
ita = 90;                % [#] pillar slope angle to the surface

groove_period_cell = round(struc_period / dx); 
groove_width_cell = round(groove_width / dx);  
pillar_depth_cell = round(pillar_depth / dy); 

% === 2. Add geometry ===
film_y_start = round(ny / 4*1);                      % [#] starting y-position
film_y_end = film_y_start + round(sum(material_thick)/dy) - 1; % [#] ending y-position
film_length = film_y_end - film_y_start + 1;
% Add multilayer to geometry
myTarget = myTarget.addMulMirror(PMLthick_cell, film_y_start, material_ids, material_thick_cell, myMaterial);
% Add grating on top of multilayer to geometry 
myTarget = myTarget.addGrating(PMLthick_cell, film_y_start, groove_period_cell, groove_width_cell, pillar_depth_cell, ita);

% === 3. Get the fields for simulation ===
[id, epr, mur, nmol, bg, mee, mhe, nVBe0, nCBe, trec, ...
    Ef, gammaCe, Gamma, wp, Omega, beta0, ...
    is_dielectric, is_metal, use_photo, use_collision, use_impact] = myTarget.getMaps();

% === 4. Check map ===
 Analysis_functions.plot_Maps(myTarget);

% === 5. Add substrate ===
eprSubstrate = mySiO2.epr;
epr(PMLthick_cell+1:end-PMLthick_cell,film_y_end+1:end) = eprSubstrate;


%% Initialize Physics Models only in Film Regions
% === Photoionization ===
tcyc    = 2;                                % take one E field data point every 'tcyc' time step for single optical cycle averge in Kelydesh rate calculation
cyc_points = round((2*pi/omega)/dt/tcyc);   % total E field data points for single optical cycle averge 
Ext = zeros(nx,ny,cyc_points);    
Eyt = zeros(nx,ny,cyc_points); 
Ezt = zeros(nx,ny,cyc_points); 

Jpxfilm = zeros(nx,film_length);
Jpyfilm = zeros(nx,film_length);
Jpzfilm = zeros(nx,film_length);

wphofilm = zeros(nx,film_length);  % [nm^-3*fs^-1] photoionization rate
logwphofilm = zeros(nx,film_length);  % [nm^-3*fs^-1] log10 base of photoionization rate
de_pop = ones(nx,film_length);    % general de-population factor for electron density

bgfilm = bg(:,film_y_start : film_y_end); 
nmolfilm = nmol(:,film_y_start : film_y_end);
nVBe0film = nVBe0(:,film_y_start : film_y_end);
nCBefilm = nCBe(:,film_y_start : film_y_end);
trecfilm = trec(:,film_y_start : film_y_end);


% === Impact Ionization ===
% Create MRE class (electron density in film[#/nm^3], grid number nx[#], film length[#], MRE highest level[#])
[MREn, MREnmax] = SavingData_functions.assignMaterialMREn(myMaterial);
myMRE = MRE(nCBefilm, nx, film_length, MREnmax);

beta0film = beta0(:,film_y_start : film_y_end); 

wcolfilm = zeros(nx,film_length);  % [fs^-1] impact ionization rate
nCBeColfilm = zeros(nx,film_length);  % [#/nm^3] electron density contributed by impact ionization
w1pt = ones(nx,film_length);        % [fs^-1] one photon absorption rate


% ===  Ohmic Heating  === 
Tefilm      = 298/11605*ones(nx,film_length); % [eV] initial electron temperature 
Tifilm      = 298/11605*ones(nx,film_length); % [eV] initial ion temperature 
Wthefilm    = 3/2*kb*11605*Tefilm;  % [J] average electron energy
Wthifilm    = 3/2*kb*11605*Tifilm;  % [J] average ion energy
Wthefilm_eff = Wthefilm;
Cefilm = ones(nx,film_length);      % [J/K/nm^3] electron heat capacity
Effilm = Ef(:,film_y_start : film_y_end);
gammaCefilm = gammaCe(:,film_y_start : film_y_end);


% === Drude Collision ===
Jdxfilm = zeros(nx,film_length);
Jdyfilm = zeros(nx,film_length);
Jdzfilm = zeros(nx,film_length);

Gammafilm = Gamma(:,film_y_start : film_y_end);
wpfilm = wp(:,film_y_start : film_y_end);
meefilm = mee(:,film_y_start : film_y_end);
mhefilm = mhe(:,film_y_start : film_y_end);
Omegafilm = Omega(:,film_y_start : film_y_end);

% === Maxwell EM Waves === 
eprx = epr(1:nx,1:ny-1);
epry = epr(1:nx-1,1:ny);
eprz = epr(2:nx-1,2:ny-1);
murx = mur(1:nx,1:ny-1);
mury = mur(1:nx-1,1:ny);
murz = mur(2:nx-1,2:ny-1);
eprfilm = epr(:,film_y_start : film_y_end);
murfilm = mur(:,film_y_start : film_y_end);

Exfilm = zeros(nx,film_length); 
Eyfilm = zeros(nx,film_length); 
Ezfilm = zeros(nx,film_length);

% === Other parameters ===
idfilm = id(:,film_y_start : film_y_end);

% Physics model flags 
is_dfilm = is_dielectric(:,film_y_start : film_y_end);
is_mfilm = is_metal(:,film_y_start : film_y_end);
is_film = is_mfilm | is_dfilm;
use_pfilm = use_photo(:,film_y_start : film_y_end);
use_cfilm = use_collision(:,film_y_start : film_y_end);
use_ifilm = use_impact(:,film_y_start : film_y_end);


%% Initialize Figures
%figHandle = figure;
%set(figHandle, 'Position', [100, 100, 2400, 1000]);

%ax1 = subplot(2, 4, 1, 'Parent', figHandle);
%ax2 = subplot(2, 4, 5, 'Parent', figHandle);
%ax3 = subplot(2, 4, 2, 'Parent', figHandle);
%ax4 = subplot(2, 4, 6, 'Parent', figHandle);
%ax5 = subplot(2, 4, 4, 'Parent', figHandle);
%ax6 = subplot(2, 4, 7, 'Parent', figHandle);
%ax7 = subplot(2, 4, 3, 'Parent', figHandle);
%ax8 = subplot(2, 4, 8, 'Parent', figHandle);

path='';

%v = VideoWriter([path,'MOVIE_',...
%    num2str(myLaser.incidenctAngle),'d_',...
%    num2str(myLaser.pulseDuration),'fs_',...
%    num2str(myLaser.spotRadius),'nm(w0)_',...
%    num2str(myLaser.wavelength),'nm(λ)_',...
%    num2str(myLaser.polarization),'-pol_',...
%    strrep(num2str(fluence), '.', 'p'),'Jcm2_',...
%    num2str(d),'nm(d)_',...
%    num2str(pillar_depth),'nm(gd)_',...  % grating depth
%    num2str(duty_cycle),'(dc)_',...  % duty cycle of groove
%    char(myMaterialNames),'.avi']);
%v.Quality=100;
%open(v);

% Open a file for writing total energy (will overwrite if it already exists)
% fid = fopen('Results/log.txt', 'w');
% E_sim_stored = zeros(Tsteps);

% Store the accumulated intensity
Emax = zeros(nx,ny);

%% Main FDTD Loop
for t = 1:Tsteps

    if strcmp(pol, 's')
        % s-pol
        Bx_r = Bx(:,1:end-1);
        By_r = By(1:end-1,:);
        Dz_r = Dz(2:end-1, 2:end-1);
        Pz_r = Pz(2:end-1, 2:end-1);

        % Update B fields
        Bx(:, 1:end-1) = k_BDx_1.*Bx(:, 1:end-1) - ...
            k_BDx_2.*(Ez(:, 2:end) - Ez(:, 1:end-1));
        By(1:end-1, :) = k_BDy_1.*By(1:end-1, :) + ...
            k_BDy_2.*(Ez(2:end, :) - Ez(1:end-1, :));
        
        % Inject E source
        Bx(:,tfsf_y-1) = Bx(:,tfsf_y-1) - k_BDx_2(:,tfsf_y-1).*Ez_TFSF(:,t);
    
        % Update H fields
        Hx(:,1:end-1) = Hx(:,1:end-1) + (k_Hx_1.*Bx(:, 1:end-1) - k_Hx_2.*Bx_r)./murx;
        Hy(1:end-1,:) = k_Hy_0.*Hy(1:end-1,:) + (k_Hy_1.*By(1:end-1, :) - k_Hy_2.*By_r)./mury;  

        % Update P fields
        Pz(2:end-1, 2:end-1) = Pz(2:end-1, 2:end-1) + (Jdz(2:end-1, 2:end-1)+Jpz(2:end-1, 2:end-1))*dt;

        % Update D fields
        Dz(2:end-1, 2:end-1) =  k_BDz_1.*Dz(2:end-1, 2:end-1) + ...
             k_BDz_2.*((Hy(2:end-1, 2:end-1) - Hy(1:end-2, 2:end-1))/dx - ...
             (Hx(2:end-1, 2:end-1) - Hx(2:end-1, 1:end-2))/dy);
    
        % Inject H source
        Dz(2:end-1,tfsf_y) = Dz(2:end-1,tfsf_y) - ...
            k_BDz_2(:,tfsf_y).*Hx_TFSF(2:end-1,t)/dy;
    
        % Update E fields
        Ez(2:end-1,2:end-1) = k_Ez_1.*Ez(2:end-1,2:end-1) + ...
            k_Ez_2.*(Dz(2:end-1,2:end-1) - Dz_r - (Pz(2:end-1,2:end-1) - Pz_r))./eprz - ...
            (Pz(2:end-1,2:end-1) - Pz_r)./(eprz*ep0);

    else
        % p-pol
        Bz_r = Bz(2:end-1, 2:end-1);
        Dx_r = Dx(:,1:end-1);
        Dy_r = Dy(1:end-1,:);
        Px_r = Px(:,1:end-1);
        Py_r = Py(1:end-1,:);

        % Update B fields
        Bz(2:end-1, 2:end-1) = k_BDz_1.*Bz(2:end-1, 2:end-1) - ...
        k_BDz_2.*((Ey(2:end-1, 2:end-1) - Ey(1:end-2, 2:end-1))/dx - ...
        (Ex(2:end-1, 2:end-1) - Ex(2:end-1, 1:end-2))/dy);
        
        % Inject E source
        Bz(2:end-1,tfsf_y) = Bz(2:end-1,tfsf_y) - ...
            k_BDz_2(:,tfsf_y).*Ex_TFSF(2:end-1,t)/dy;
    
        % Update H fields
        Hz(2:end-1,2:end-1) = k_Hz_1.*Hz(2:end-1,2:end-1) + k_Hz_2.*(Bz(2:end-1,2:end-1)-Bz_r)./murz;

        % Update P fields
        Px(:,1:end-1) = Px(:,1:end-1) + (Jdx(:,1:end-1)+Jpx(:,1:end-1))*dt;
        Py(1:end-1,:) = Py(1:end-1,:) + (Jdy(1:end-1,:)+Jpy(1:end-1,:))*dt;

        % Update D fields
        Dx(:, 1:end-1) = k_BDx_1.*Dx(:, 1:end-1) + k_BDx_2.*(Hz(:, 2:end) - Hz(:, 1:end-1));
        Dy(1:end-1, :) = k_BDy_1.*Dy(1:end-1, :) - k_BDy_2.*(Hz(2:end, :) - Hz(1:end-1, :));
    
        % Inject H source
        Dx(:,tfsf_y-1) = Dx(:,tfsf_y-1) - k_BDx_2(:,tfsf_y-1).*Hz_TFSF(:,t);
    
        % Update E fields
        Ex(:,1:end-1) = Ex(:,1:end-1) + (k_Ex_1.*Dx(:,1:end-1) - k_Ex_2.*Dx_r)./eprx - ...
            (Px(:,1:end-1) - Px_r)./(eprx*ep0);
        Ey(1:end-1,:) = k_Ey_0.*Ey(1:end-1,:) + (k_Ey_1.*Dy(1:end-1,:) - k_Ey_2.*Dy_r)./epry - ...
            (Py(1:end-1,:) - Py_r)./(epry*ep0);   
    end


    % recording index of E for a single cycle
    Index = mod(t,cyc_points);

    Ext(:,:,Index+1) = Ex;
    Eyt(:,:,Index+1) = Ey;
    Ezt(:,:,Index+1) = Ez;

    Exfilm_r = Exfilm; 
    Eyfilm_r = Eyfilm; 
    Ezfilm_r = Ezfilm; 
    Exfilm = Ex(:,film_y_start:film_y_end); 
    Eyfilm = Ey(:,film_y_start:film_y_end); 
    Ezfilm = Ez(:,film_y_start:film_y_end); 


    % Store cycle averaged envolope Eenv and accumulated Emax
    Eampt = sqrt(Ext.^2 + Eyt.^2 + Ezt.^2);
    Eenv = max(Eampt,[],3);     % sqrt of single-cycle time maximum (V/nm) 
    Eenvfilm = Eenv(:,film_y_start:film_y_end);

    E = sqrt(Ex.^2+Ey.^2+Ez.^2);
    Efilm = E(:,film_y_start:film_y_end);
    maskEmax = E>Emax;
    Emax(maskEmax) = E(maskEmax);


    %% Update CB electron density 
    nCBefilm_r = nCBefilm;
    nCBeColfilm_r = nCBeColfilm;
    nCBefilmmax = myMRE.nCBefilmmax;  % seed electron density for impact ionization
    nCBeColfilm(is_dfilm) = dt*nCBefilmmax(is_dfilm).*wcolfilm(is_dfilm);

    nCBefilm(is_dfilm) = nCBefilm(is_dfilm) + ...
        dt*de_pop(is_dfilm).*wphofilm(is_dfilm) + ...
        nCBeColfilm(is_dfilm);

    % update the MRE for impact ionization
    myMRE = myMRE.updateMRE(nCBefilm, nCBefilm_r, is_dfilm, dt, wcolfilm, trecfilm, w1pt);
    % build combined "max-level" population
    myMRE = myMRE.updateMREnCBefilmmax(idfilm, MREn);

    nCBefilm(is_dfilm) = nCBefilm(is_dfilm) ...
    - dt*nCBefilm(is_dfilm)./trecfilm(is_dfilm); % recombination is considered in MRE


    %% Update effective Jp current density:
    logEenvfilm = log10(Eenvfilm)+9;  % [V/m] log10 base
    gr = logEenvfilm>1 & is_dfilm;
    grbg = Eenvfilm*10^9>1.585e+03 & is_dfilm;   % [V/m] 

    % *10^12 to convert [J] in terms of [fs] and [nm]
    Jpxfilm(gr) = sign(Exfilm(gr)).*(bgfilm(gr)*10^12.*wphofilm(gr).*de_pop(gr)./Eenvfilm(gr));     
    Jpxfilm(~gr) = 0;
    Jpyfilm(gr) = sign(Eyfilm(gr)).*(bgfilm(gr)*10^12.*wphofilm(gr).*de_pop(gr)./Eenvfilm(gr));      
    Jpyfilm(~gr) = 0;
    Jpzfilm(gr) = sign(Ezfilm(gr)).*(bgfilm(gr)*10^12.*wphofilm(gr).*de_pop(gr)./Eenvfilm(gr));      
    Jpzfilm(~gr) = 0;
    
    Jpx(:,film_y_start:film_y_end) = Jpxfilm;
    Jpy(:,film_y_start:film_y_end) = Jpyfilm;    
    Jpz(:,film_y_start:film_y_end) = Jpzfilm;


    %% Update Keldysh photoionization rate
    bgfilm_r = bgfilm;
    for i = 1:(length(myMaterial) - numMetals)
        logwphofilm_i = zeros(nx,film_length);
        logwphofilm_i(gr) = interp1(myMaterial(i).log10E_Keldysh,myMaterial(i).log10wpho_Keldysh,logEenvfilm(gr));
        logwphofilm(idfilm == i) = logwphofilm_i(idfilm == i);

        bgfilm_i = bgfilm;
        bgfilm_i(grbg) = interp1(myMaterial(i).E_Keldysh,myMaterial(i).bgeff_Keldysh,Eenvfilm(grbg)*10^9);  
        bgfilm(idfilm == i) = bgfilm_i(idfilm == i);  % [J]
    end
    logwphofilm(~gr) = -Inf;
    wphofilm = 10.^(logwphofilm-21); % [fs^-1*nm^-3]


    %% Update impact photoionization rate
    wcolfilm(is_dfilm) = Physics_functions.updateImpactIonizationRate_Keldysh(Wthefilm, meefilm, mhefilm, bgfilm, beta0film, de_pop, eprfilm, is_dfilm);
%     wcolfilm(is_dfilm) = Physics_functions.updateImpactIonizationRate_Drude(Efilm, omega, eprfilm, bgfilm, meefilm, Gammafilm, is_dfilm);


    %% Update Ohmic heating
    JExfilm = Jdxfilm.*(Exfilm+Exfilm_r)/2;
    JEyfilm = Jdyfilm.*(Eyfilm+Eyfilm_r)/2;
    JEzfilm = Jdzfilm.*(Ezfilm+Ezfilm_r)/2;
    Q = dt*abs(JExfilm+JEyfilm+JEzfilm);
    Q(is_dfilm) = Q(is_dfilm) - bgfilm_r(is_dfilm).*beta0film(is_dfilm).*(nCBeColfilm(is_dfilm)+nCBeColfilm_r(is_dfilm))/2;  % subtract the heat used by impact ionization

    Wthefilm_r = Wthefilm;
    Wthefilm_r(is_dfilm) = Tefilm(is_dfilm)*(3/2*kb)*11605; % [J]
    Wthefilm_r(is_mfilm) = (Tefilm(is_mfilm)*11605).^2.*gammaCefilm(is_mfilm)/2; % [J]
    Tefilm_r = Tefilm;

    Wthefilm(is_film) = (Q(is_film) + ...
        nCBefilm_r(is_film) .* Wthefilm(is_film)*1e12) ./ ...
        nCBefilm(is_film)/1e12;  % [J] Energy per electron, or average energy. *1e12 to convert J in terms of [fs] and [nm], then /1e12 to convert back to SI.
    
    % update electron temperature for dielectric (ideal gas) and metal (fermi gas)
    Tefilm(is_dfilm) = Wthefilm(is_dfilm) / (3/2*kb) / 11605;  % [eV]
    Tefilm(is_mfilm) = sqrt(2*Wthefilm(is_mfilm) ./ gammaCefilm(is_mfilm))/ 11605;  % [eV]
    
    % update electron temperature diffusion
    Cefilm(is_mfilm) = gammaCefilm(is_mfilm).*Tefilm(is_mfilm).*nCBefilm_r(is_mfilm)*10^27*11605;  % [J/K/m^3] electron heat capacity
    Tefilm(is_mfilm) = Physics_functions.updateTeDiffusion_metal(Tefilm(is_mfilm), Cefilm(is_mfilm), dt, d);
    
    % update one photon absorption rate
    w1pt(is_dfilm) = Q(is_dfilm)/(dt*10^-15)/1e12./nCBefilm(is_dfilm)/ (1e15*1.054e-34*omega)*10^-15;  % [1/fs]


    %% Update Jd currents
    % Trapezoidal Rule (semi-implicit, stable)
    [Jdxfilm, Jdyfilm, Jdzfilm] = Physics_functions.updateDrudeCurrent_trapezoidal( ...
    Jdxfilm, Jdyfilm, Jdzfilm, Exfilm, Eyfilm, Ezfilm, ...
    Gammafilm, wpfilm, dt);
%     % Explicit Euler scheme
%     [Jdxfilm, Jdyfilm, Jdzfilm] = Physics_functions.updateDrudeCurrent_euler( ...
%     Jdxfilm, Jdyfilm, Jdzfilm, Exfilm, Eyfilm, Ezfilm, ...
%     Gammafilm, wpfilm, dt);

    Jdx(:,film_y_start:film_y_end) = Jdxfilm;
    Jdy(:,film_y_start:film_y_end) = Jdyfilm;    
    Jdz(:,film_y_start:film_y_end) = Jdzfilm;

    wpfilm(is_dfilm) = sqrt(nCBefilm_r(is_dfilm) * e^2 ./ (ep0 * me * meefilm(is_dfilm)));


    %% Update collision rate
    Gammafilm(is_dfilm) = Physics_functions.updateCollisionRate_dielectric(nCBefilm_r, Tefilm_r, Wthefilm_r, meefilm, nmolfilm, is_dfilm);


    %% Update de-population factor
    de_pop(is_dfilm) = (nVBe0film(is_dfilm) - 2*nCBefilm(is_dfilm))./(nVBe0film(is_dfilm) - ...
        nCBefilm(is_dfilm));


    %% Update refractive index
    eprfilm(is_dfilm) = 1 + (nmolfilm(is_dfilm)-nCBefilm(is_dfilm))/10^(-3*9)*e^2 ./ ...
        ((ep0*10^-3)*meefilm(is_dfilm)*me)/10^30./Omegafilm(is_dfilm).^2;
    epr(:,film_y_start : film_y_end) = eprfilm;
    eprx = epr(1:nx,1:ny-1);
    epry = epr(1:nx-1,1:ny);
    eprz = epr(2:nx-1,2:ny-1);


    %% Post Update
%     % Energy conservation check
%     [E_sim, E_theory] = Analysis_functions.calc_Total_Energy(Ex, Ey, Ez, Hx, Hy, Hz, myGrid, myLaser, epr);
%     % Write the energy data to the file
%     fprintf(fid, 't = %.2f fs | E_sim = %.4e J/m | E_theory = %.4e J/m | Efficiency = %.2f%%\n', ...
%         t * dt, E_sim, E_theory, 100 * E_sim / E_theory);
%     E_sim_stored(t) = E_sim; 
        
    % Visualization and Output
%    if mod(t,300) == 0        
%        Analysis_functions.plot_STEP4_12nm_new(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, myLaser, myGrid, myMaterialNames, id, ...
%            t, tfsf_y, film_y_start, film_y_end,...
%            E, Emax, nCBefilm, Tefilm, wcolfilm, wphofilm, Gammafilm);
%        frame=getframe(figHandle);
%        writeVideo(v,frame);
%    end

    % Save data
    if mod(t,1200) == 0
    csvwrite([path,'ne_nm^3_',num2str(dt*t),'fs.csv'],nCBefilm);
    csvwrite([path,'Emax_Vnm_',num2str(dt*t),'fs.csv'],Emax);
    %csvwrite([path,'wcol_fs^-1_',num2str(dt*t),'fs.csv'],wcolfilm);
    %csvwrite([path,'wpho_fs^-1_',num2str(dt*t),'fs.csv'],wphofilm);
    %csvwrite([path,'Gamma_fs^-1_',num2str(dt*t),'fs.csv'],Gammafilm);
    %csvwrite([path,'Te_eV_fs^-1_',num2str(dt*t),'fs.csv'],Tefilm);
    end

end

% Close the video file
% close(v);
% Close the txt file
% fclose(fid);

% Save data
% savefile = 0;
% if savefile == 1
%    csvwrite([path,'id.csv'], idfilm);
%    csvwrite([path,'epr.csv'], eprfilm);
%    csvwrite([path,'Emax_Vnm.csv'],Emax);
%    csvwrite([path,'Te_eV.csv'],Tefilm);
%    csvwrite([path,'ne_nm^3.csv'], nCBefilm);
% end


% =========================================================
% Save results for Python to read back
% =========================================================
fdtd_result = table(aoi, pillar_depth, 1 - duty_cycle, max(max(nCBefilm)), ...
    'VariableNames', {'aoi', 'tp', 'dc', 'ne_peak'});
writetable(fdtd_result, '<result_fdtd>');