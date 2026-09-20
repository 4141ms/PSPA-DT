function sim_recon_fun(model, dataset, input_root, output_root)
%SIM_RECON_FUN Run k-Wave forward simulation and time-reversal reconstruction.
%
% Inputs:  <input_root>/<model>/<dataset>{,_p0}
% Outputs: <output_root>/<model>/<dataset>_{recon,high,dog,fft}
% output_root defaults to input_root.

arguments
    model (1,1) string
    dataset (1,1) string
    input_root (1,1) string
    output_root (1,1) string = ""
end

if strlength(output_root) == 0
    output_root = input_root;
end

phantom_dir = fullfile(input_root, model, dataset);
mu_dir = fullfile(input_root, model, dataset + "_p0");
recon_dir = fullfile(output_root, model, dataset + "_recon");
v_high = fullfile(output_root, model, dataset + "_high");
v_dog = fullfile(output_root, model, dataset + "_dog");
v_fft = fullfile(output_root, model, dataset + "_fft");

if ~isfolder(phantom_dir)
    error("Input directory does not exist: %s", phantom_dir);
end
if ~isfolder(mu_dir)
    error("Initial-pressure directory does not exist: %s", mu_dir);
end

if ~exist(recon_dir, 'dir')
    mkdir(recon_dir);
end

if ~exist(v_high, 'dir')
    mkdir(v_high);
end

if ~exist(v_dog, 'dir')
    mkdir(v_dog);
end

if ~exist(v_fft, 'dir')
    mkdir(v_fft);
end

if model == "gt"
    file_list = dir(fullfile(phantom_dir, "IXI*"));
else
    file_list = dir(fullfile(phantom_dir, "*.nii.gz"));
end

len = numel(file_list);
fprintf("Found %d nii.gz files in %s\n", len, phantom_dir);

%%
%Grid setting
Nx = 300;
Ny = Nx;
Nz = 130;
PML_size = getOptimalPMLSize([Nx, Ny, Nz]);

x = 300e-3;  % [m]
y = x;  % [m]
z = 135e-3;  % [m]
dx = x/Nx;
dy = y/Ny;
dz = z/Nz;

kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
% Sample setting
sample_num = 3000;
T = 240e-6; % [s]
dt = T / sample_num;
kgrid.t_array = 0:dt:T-dt;

pad = (Nx - 256)/2; 

% Acoustic properties setting (1-speed of sound; 2-density)
acoustic_prop = [randi([1532,1573]), randi([1040,1043]);    % White-Matter
                 randi([1532,1573]), randi([1039,1050]);    % Gray-Matter
                 randi([1502,1507]), randi([1007,1007]);    % CSF
                 randi([1545,1631]), randi([1041,1178]);    % Muscle
                 randi([1537,1720]), randi([1100,1125]);    % Scalp
                 randi([1495,1544]), randi([1000,1009]);    % Eye_balls
                 randi([2660,4200]), randi([1800,2100]);    % Compact_bone
                 randi([1854,2450]), randi([1080,1350]);    % Spongy_bone
                 randi([1559,1590]), randi([1025,1060]);    % Vessel
                ];

for f = 1:len
    if model == "gt"
        nii_path = fullfile(file_list(f).folder, file_list(f).name, "seg.nii.gz");
        name = file_list(f).name;
    else
        nii_path = fullfile(file_list(f).folder, file_list(f).name);
        [~, base, ext] = fileparts(nii_path);
        if strcmpi(ext, ".gz")
            [~, name] = fileparts(base);   % base: "name.nii" -> name: "name"
        else
            name = base;
        end
    end
    
    

    p0_path = fullfile(mu_dir, string(name) + "_mu.mat");

    info = niftiinfo(nii_path);
    phantom = single(niftiread(info));
    [nx, ny, nz] = size(phantom);
    load(p0_path); %% 


    source = struct();
    source.p0 = zeros(Nx,Ny,Nz,'single');
    source.p0(pad:(pad+nx-1), pad:(pad+ny-1), 1:nz) = p0;    
    

    medium.sound_speed = 1482 * ones(Nx,Ny,Nz);
    medium.density = 994 * ones(Nx,Ny,Nz);

    ph_sos = 1482 * ones(nx,ny,nz);
    ph_den = 994 * ones(nx,ny,nz);
    for i = 1:9
        ph_sos(phantom == i) = acoustic_prop(i, 1);
        ph_den(phantom == i) = acoustic_prop(i, 2);
    end
    
    medium.sound_speed(pad:(pad+nx-1), pad:(pad+ny-1), 1:nz) = ph_sos;
    medium.density(pad:(pad+nx-1), pad:(pad+ny-1), 1:nz) = ph_den;

    Ns_target = 2048;

    margin_grid = 12;
    margin_m = margin_grid * dx;

    Lx = Nx*dx; Ly = Ny*dy; Lz = Nz*dz;
    Rmax = max([Lx, Ly, Lz]) / 2 - margin_m;
    
    if Rmax <= 0
        error('grid is too small');
    end
    

    R = 0.95 * Rmax;
    
    fprintf('Grid physical size (mm): Lx=%.1f, Ly=%.1f, Lz=%.1f\n', Lx*1e3, Ly*1e3, Lz*1e3);
    fprintf('Auto Rmax=%.2f mm, using R=%.2f mm\n', Rmax*1e3, R*1e3);
    
    zc = +Lz/2;    
    center_xyz_m = [0, 0, -zc];
    
    % --- Fibonacci  ---
    i = (0:Ns_target-1)';
    phi = acos(1 - 2*(i+0.5)/Ns_target);
    theta = 2*pi*(i+0.5) / ((1 + sqrt(5))/2);
    
    xu = sin(phi).*cos(theta);
    yu = sin(phi).*sin(theta);
    zu = cos(phi);
    
    keep = zu >= 0;                 % 半球：x >= 0
    xu = xu(keep); yu = yu(keep); zu = zu(keep);
    
    x = center_xyz_m(1) + R*xu;
    y = center_xyz_m(2) + R*yu;
    z = center_xyz_m(3) + R*zu;
    
    sensor_cart = [x.'; y.'; z.'];    % 3 x N (meters)
    sensor = struct();
    [sensor.mask, order_index] = cart2grid(kgrid, sensor_cart);
    sensor.record = {'p'};
    
    Ns = numel(order_index);
    fprintf('Ns (order_index) = %d\n', Ns);
    sensor_cart_ord = sensor_cart(:, order_index);   % 3 x Ns_eff (meters)
    
    Ns_eff = size(sensor_cart_ord, 2);
    fprintf('Requested points ~%d, Ns_eff=%d (after cart2grid)\n', size(sensor_cart,2), Ns_eff);

    %% -----------------------------
    % (6) Forward simulation
    % -----------------------------
    input_args = { ...
        'PMLInside', false, ...
        'DataCast', 'single', ...
        'PlotSim', false ...
    };
    
    sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    % -----------------------------
    % (7) Time reversal reconstruction
    % -----------------------------
    source_tr = struct();
    source_tr.p0 = 0;
    
    sensor_tr = struct(); % no p0 for TR
    sensor_tr.mask = sensor.mask;
    
    sensor_tr.time_reversal_boundary_data = sensor_data.p;
    sensor_tr.record = {'p_final'};
    
    recon_out   = kspaceFirstOrder3DG(kgrid, medium, source_tr, sensor_tr, input_args{:});
    if isstruct(recon_out)
        recon_xyz = recon_out.p_final;
    else
        recon_xyz = recon_out;   
    end

    recon = recon_xyz;
    recon_path = fullfile(recon_dir, string(name) + "-V.mat");
    save(recon_path, 'recon');
    %% high pass
    V = recon_xyz;                         % Nx×Ny×Nz
    V = V - mean(V(:));                    % 
    
    sigma = 2;                             % 
    V_low  = imgaussfilt3(V, sigma);
    V_high = V - V_low;                    % 
    
    high_path = fullfile(v_high, string(name) + "-high.mat");
    save(high_path, 'V_high');

    %% dog
    s1 = 1.0;
    s2 = 4.0;                              % s2 > s1
    V1 = imgaussfilt3(V, s1);
    V2 = imgaussfilt3(V, s2);
    V_dog = V1 - V2;                     


    dog_path = fullfile(v_dog, string(name) + "-dog.mat");
    save(dog_path, 'V_dog');

    %% fft
    V = recon_xyz;
    V = V - mean(V(:));
    
    [Nx,Ny,Nz] = size(V);
    Fx = (-floor(Nx/2):ceil(Nx/2)-1)/Nx;
    Fy = (-floor(Ny/2):ceil(Ny/2)-1)/Ny;
    Fz = (-floor(Nz/2):ceil(Nz/2)-1)/Nz;
    [fy,fx,fz] = ndgrid(Fy,Fx,Fz);         
    
    R = sqrt(fx.^2 + fy.^2 + fz.^2);      
    fc = 0.05;                             
    n  = 4;                                
    
    H = 1 ./ (1 + (fc ./ (R + eps)).^(2*n));  % Butterworth
    
    Vf = fftshift(fftn(V));
    Vhp = real(ifftn(ifftshift(Vf .* H)));
    
    fft_path = fullfile(v_fft, string(name) + "-fft.mat");
    save(fft_path, 'Vhp');
    % break
end
end
