clear; clc; close all;


%% Setup paths
projectRoot = '/fs/ess/PAS1730/RCWA';
addpath(genpath(fullfile(projectRoot, 'RCWA')));

% =========================================================
% Multilayer grating
% units: um
%
% Top incidence
% =========================================================

% -------------------------
% optical parameters
% -------------------------
LD0 = 1.95;       % center wavelength in um
teta0 = <aoi>	;      % incident angle in degree

nAir = 1.00;
nL   = 1.45;       % SiO2
nH   = 1.922;       % HfO2
nSub = 1.45;       % substrate if needed

beta0 = nAir * sind(teta0);

% -------------------------
% geometry from paper
% -------------------------
D   = 1.587301587;       % period, um
h   = <tp>/1000;       % grating pillar height, um
w_b = D*<dc>;       % pillar bottom width, um

e   = 1.475-h;       % residual thickness, um
phi = 90;        % sidewall angle, measured from horizontal/base plane

% linewidth at top
dx_one_side = h / tand(phi);
w_t = w_b - 2*dx_one_side;

fprintf('Period D          = %.6f um\n', D);
fprintf('Bottom width w_b  = %.6f um\n', w_b);
fprintf('Top width w_t     = %.6f um\n', w_t);
fprintf('Residual e        = %.6f um\n', e);
fprintf('Pillar height h   = %.6f um\n', h);

if w_t <= 0
    error('Top width <= 0. Check phi, h, and w_b.');
end

% RCWA settings
nn = 21;               % Fourier orders
Nslices = 1;          % staircase slices for sloped sidewall
slice_h = h / Nslices;

% =========================================================
% textures
% =========================================================
% 1 = air
% 2 = SiO2
% 3 = HfO2
% 4 = substrate
% 5... = patterned SiO2 slices in air

textures = cell(1, 4 + Nslices);
textures{1} = nAir;
textures{2} = nL;
textures{3} = nH;
textures{4} = nSub;

% trapezoidal SiO2 grating slices, top -> bottom
for j = 1:Nslices
    frac = (j - 0.5) / Nslices;
    wj = w_t + frac * (w_b - w_t);   % width grows from top to bottom

    x1 = -wj/2;
    x2 =  wj/2;

    % SiO2 ridge in air
    textures{4 + j} = { [x1, x2], [nAir, nL] };
end

% from second layer of top to bottom
layers_nm = [
    178.4535,3;
    495.4981,2;
    178.4535,3;
    495.4981,2;
    267.6802,3;
    409.238286,2;
    266.748672,3;
    346.324261,2;
    264.145669,3;
    381.206798,2;
    250.325632,3;
    379.765297,2;
    266.371252,3;
    356.99416,2;
    266.001111,3;
    381.822588,2;
    253.697616,3;
    374.437941,2;
    265.410292,3;
    363.067091,2;
    267.277743,3;
    373.324345,2;
    256.458726,3;
    381.928003,2;
    265.575303,3;
    371.6235,2;
];

% -------------------------
% initialize solver
% -------------------------
pol = 1;                    % 1 = TE, -1 = TM
parm = res0(pol);
parm.not_io = 1;
parm.res1.trace = 0;

% =========================================================
% profile construction
% =========================================================
hauteurs = [];
sequence = [];

% air
hauteurs = [hauteurs, 1];
sequence = [sequence, 1];

% grating slices from top to bottom
for j = 1:Nslices
    hauteurs = [hauteurs, slice_h];
    sequence = [sequence, 4+j];
end

% residual SiO2
hauteurs = [hauteurs, e];
sequence = [sequence, 2];

% add layers
for i = 1:size(layers_nm,1)
    thickness_um = layers_nm(i,1) / 1000;   % nm → um
    material     = layers_nm(i,2);
    
    hauteurs = [hauteurs, thickness_um];
    sequence = [sequence, material];
end

% substrate
hauteurs = [hauteurs, 1];
sequence = [sequence, 4];

profil = {hauteurs, sequence};

% disp(' ');
% disp('Heights (um):');
% disp(hauteurs);
% 
% disp('Sequence:');
% disp(sequence);

% =========================================================
% Single-wavelength geometry and field calculation at LD0
% =========================================================

LD = LD0;

aa = res1(LD, D, textures, nn, beta0, parm);

% % =========================================================
% % geometry-only plot
% % =========================================================
x = linspace(-D/2, D/2, 401);

parm_plot = res0(pol);
parm_plot.not_io = 1;
parm_plot.res3.cale = [];
parm_plot.res3.trace = 1;

[tab_geo, z_geo, o_geo] = res3(x, aa, profil, [1,1], parm_plot);

% =========================================================
% diffraction / reflection / transmission at LD0
% =========================================================
parm2 = res0(pol);
parm2.not_io = 1;

ef = res2(aa, profil, parm2);

R = sum(ef.inc_top_reflected.efficiency);
T = sum(ef.inc_top_transmitted.efficiency);
A = 1 - R - T;

disp(' ');
disp('incident from top')
disp(rettexte('top:',    ef.inc_top_reflected.efficiency));
disp(rettexte('bottom:', ef.inc_top_transmitted.efficiency));
fprintf('At LD = %.6f um:\n', LD);
fprintf('R = %.8f\n', R);
fprintf('T = %.8f\n', T);
fprintf('A = %.8f\n', A);
fprintf('R+T+A = %.8f\n', R+T+A);

% =========================================================
% top-incidence reflected propagating orders at LD0
% =========================================================
theta_r = ef.inc_top_reflected.theta(:);
eff_r   = ef.inc_top_reflected.efficiency(:);

m_r = round((sind(theta_r) - sind(teta0)) * D / LD);
theta_calc_r = asind(sind(teta0) + m_r * LD / D);

disp(' ');
disp('Top incidence -> reflected propagating orders');
disp(table(m_r, theta_calc_r, theta_r, eff_r, ...
    'VariableNames', {'Order','Theta_calc_deg','Theta_solver_deg','Efficiency'}));

% =========================================================
% top-incidence transmitted propagating orders at LD0
% =========================================================
theta_t = ef.inc_top_transmitted.theta(:);
eff_t   = ef.inc_top_transmitted.efficiency(:);

m_t = round((nSub*sind(theta_t) - nAir*sind(teta0)) * D / LD);
theta_calc_t = asind((nAir*sind(teta0) + m_t * LD / D) / nSub);

disp(' ');
disp('Top incidence -> transmitted propagating orders');
disp(table(m_t, theta_calc_t, theta_t, eff_t, ...
    'VariableNames', {'Order','Theta_calc_deg','Theta_solver_deg','Efficiency'}));

% =========================================================
% field calculation: top incidence at LD0
% =========================================================
parm_field = res0(pol);
parm_field.not_io = 1;
parm_field.res3.sens = 1;
parm_field.res3.trace = 1;
parm_field.res3.champs = [1,2,3];

[e_top, z_top, o_top] = res3(x, aa, profil, 1, parm_field);

comp1_top = e_top(:,:,1);
comp2_top = e_top(:,:,2);
comp3_top = e_top(:,:,3);

% figure;
% imagesc(x, z_top, log10(abs(comp1_top) + 1e-12));
% axis xy;
% xlabel('x (\mum)');
% ylabel('z (\mum)');
% title('log10 |Component 1|, top incidence');
% colorbar;
% 
% figure;
% imagesc(x, z_top, log10(abs(comp2_top) + 1e-12));
% axis xy;
% xlabel('x (\mum)');
% ylabel('z (\mum)');
% title('log10 |Component 2|, top incidence');
% colorbar;
% 
% figure;
% imagesc(x, z_top, log10(abs(comp3_top) + 1e-12));
% axis xy;
% xlabel('x (\mum)');
% ylabel('z (\mum)');
% title('log10 |Component 3|, top incidence');
% colorbar;

% =========================================================
% component labels
% =========================================================
% if pol == 1
%     disp(' ');
%     disp('Polarization = TE');
%     disp('Component 1 = Ey');
%     disp('Component 2 = Hx');
%     disp('Component 3 = Hz');
% else
%     disp(' ');
%     disp('Polarization = TM');
%     disp('Component 1 = Hy');
%     disp('Component 2 = Ex');
%     disp('Component 3 = Ez');
% end

% =========================================================
% Wavelength sweep: DE vs wavelength
% =========================================================

LD_list = linspace(1.850, 2.050, 20);   % um

DE_m1 = zeros(size(LD_list));            % reflected -1 order
DE_0  = zeros(size(LD_list));            % reflected 0 order
DE_p1 = zeros(size(LD_list));            % reflected +1 order

R_all = zeros(size(LD_list));
T_all = zeros(size(LD_list));
A_all = zeros(size(LD_list));

for iw = 1:length(LD_list)

    LD = LD_list(iw);

    % Recalculate RCWA solution at each wavelength
    aa_w = res1(LD, D, textures, nn, beta0, parm);

    ef_w = res2(aa_w, profil, parm2);

    % Total reflection, transmission, absorption
    R_all(iw) = sum(ef_w.inc_top_reflected.efficiency);
    T_all(iw) = sum(ef_w.inc_top_transmitted.efficiency);
    A_all(iw) = 1 - R_all(iw) - T_all(iw);

    % Reflected propagating orders
    theta_r_w = ef_w.inc_top_reflected.theta(:);
    eff_r_w   = ef_w.inc_top_reflected.efficiency(:);

    % Diffraction order index
    m_r_w = round((sind(theta_r_w) - sind(teta0)) * D / LD);

    % Extract reflected -1 order
    idx_m1 = find(m_r_w == -1, 1);
    if ~isempty(idx_m1)
        DE_m1(iw) = eff_r_w(idx_m1);
    else
        DE_m1(iw) = 0;
    end

    % Extract reflected 0 order
    idx_0 = find(m_r_w == 0, 1);
    if ~isempty(idx_0)
        DE_0(iw) = eff_r_w(idx_0);
    else
        DE_0(iw) = 0;
    end

    % Extract reflected +1 order
    idx_p1 = find(m_r_w == 1, 1);
    if ~isempty(idx_p1)
        DE_p1(iw) = eff_r_w(idx_p1);
    else
        DE_p1(iw) = 0;
    end

    fprintf('LD = %.4f um, R = %.6f, T = %.6f, A = %.6f, DE(-1) = %.6f\n', ...
        LD, R_all(iw), T_all(iw), A_all(iw), DE_m1(iw));

end

% =========================================================
% Plot reflected DE vs wavelength
% =========================================================

figure;
plot(LD_list, DE_m1, 'LineWidth', 3);
hold on;
% plot(LD_list, DE_0, 'LineWidth', 2);
% plot(LD_list, DE_p1, 'LineWidth', 2);
plot(LD_list, R_all, '--', 'LineWidth', 1.5);

xlabel('Wavelength (\mum)');
ylabel('Diffraction efficiency');
legend('Reflected -1 order', ...  %  'Reflected 0 order', ... 'Reflected +1 order', ...
       'Total R', ...
       'Location', 'best');

title('Reflected Diffraction Efficiency vs Wavelength');
grid on;
% xline(LD0, ':', '1.95 \mum');
exportgraphics(gcf, 'DE.png', 'Resolution', 300);

% =========================================================
% Save sweep results
% =========================================================

sweep_result = table(LD_list(:), DE_m1(:), ...
                     R_all(:), T_all(:), ...
    'VariableNames', {'Wavelength_um', ...
                      'DE_reflected_m1', ...
                      'R_total', ...
                      'T_total', ...
});

writetable(sweep_result, '<DE_filename>');

disp(' ');
disp('Saved wavelength sweep result to DE_vs_wavelength.csv');


% =========================================================
% Save results for Python to read back
% =========================================================
rcwa_result = table(teta0, <tp>, <dc>, max(DE_m1), mean(DE_m1), ...
    'VariableNames', {'aoi', 'tp', 'dc', 'DE_m1_peak', 'DE_m1_avg'});
writetable(rcwa_result, '<result_rcwa>');