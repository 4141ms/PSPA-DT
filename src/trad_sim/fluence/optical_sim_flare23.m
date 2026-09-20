function optical_sim_flare23(seg_root, output_root, wavelength, gpu_id, random_seed)
%OPTICAL_SIM_FLARE23 Generate FLARE23 photon-flux targets with MCX.
%
% optical_sim_flare23(seg_root, output_root, wavelength, gpu_id, random_seed)
% wavelength must be "700nm" or "1064nm". Generated data are stored below
% <output_root>/<wavelength>. The random seed controls the per-run optical
% property perturbations.

arguments
    seg_root (1,1) string
    output_root (1,1) string
    wavelength (1,1) string {mustBeMember(wavelength,["700nm","1064nm"])} = "1064nm"
    gpu_id (1,1) string = "1"
    random_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 0
end

if ~isfolder(seg_root)
    error("Input directory does not exist: %s", seg_root);
end
rng(random_seed, "twister");

process_data_path = fullfile(output_root, wavelength, "ProcessFLARE23");
flux_log_path = fullfile(output_root, wavelength, "brain_log_flux_3d");
flux_path = fullfile(output_root, wavelength, "brain_flux_3d");
press_path = fullfile(output_root, wavelength, "brain_init_press");
mua_path = fullfile(output_root, wavelength, "brain_mua");


if ~exist(mua_path, 'dir')
    mkdir(mua_path);
end
if ~exist(process_data_path, 'dir')
    mkdir(process_data_path);
end
if ~exist(flux_log_path, 'dir')
    mkdir(flux_log_path);
end
if ~exist(flux_path, 'dir')
    mkdir(flux_path);
end
if ~exist(press_path, 'dir')
    mkdir(press_path);
end

% Load file list
files = dir(fullfile(seg_root, "**", "seg.nii.gz"));
ct_files = dir(fullfile(seg_root, "**", "ct.nii.gz"));
if numel(files) ~= numel(ct_files)
    error("Found %d segmentation files but %d CT files under %s.", ...
        numel(files), numel(ct_files), seg_root);
end

for index = 1:length(files)
% for index = 1:1
    % get file absolute path
    seg_path = fullfile(files(index).folder, files(index).name);
    ct_path = fullfile(ct_files(index).folder, ct_files(index).name);
    % get the name of folder which contains the file(FLARE23-xxxx)
    [~, parent_folder] = fileparts(files(index).folder);
    case_name = string(parent_folder);
    disp(case_name);
    [seg_processed, ct] = processed_ct_seg(seg_path, ct_path);
    [seg_padded, rows_in_padded, cols_in_padded] = symmetry_pad(seg_processed);
    seg = seg_padded;

    % data = niftiread(seg_path); 
    % info = niftiinfo(seg_path); 
    % 
    % seg = data;
    [Nx, Ny, depth] = size(seg);
    seg_3d = zeros(Nx, Ny, depth);
    seg_3d(:, :, 1:depth) = seg(:,:, end - depth + 1:end);
    cfg.vol = uint8(seg_3d);


    % [mu_a (mm^-1), mu_s' (mm^-1), g, n]
    if wavelength == "700nm"
        % Optical properties of abdominal tissues at 700 nm
        % opt_prop = [mua, mus, g, n]
        
        opt_prop = [
            0       0       1      1.00;    % 0 Background (Air)
    
            0.22    24.0    0.90   1.40;    % 1 Liver, blood-rich parenchymal organ
            0.12    18.5    0.91   1.39;    % 2 Right kidney
            0.32    27.0    0.91   1.40;    % 3 Spleen, highly blood-rich organ
            0.07    12.0    0.92   1.38;    % 4 Pancreas

            0.126   30.75   0.989  1.40;    % 5 Aorta, approximated as blood vessel
            0.115   28.0    0.989  1.37;    % 6 Inferior vena cava, venous blood vessel
    
            0.08    14.0    0.90   1.42;    % 7 Right adrenal gland
            0.08    14.0    0.90   1.42;    % 8 Left adrenal gland
    
            0.012   4.0     0.95   1.35;    % 9 Gallbladder, bile/fluid dominated
            0.075   11.0    0.90   1.38;    % 10 Esophagus
            0.085   13.0    0.91   1.38;    % 11 Stomach
            0.085   13.0    0.91   1.38;    % 12 Duodenum
    
            0.12    18.5    0.91   1.39;    % 13 Left kidney
    
            0.0187  12.24   0.89   1.43;    % 14 Bone
            0.0005  11.0    0.80   1.45;    % 15 Fat
            0.0536  21.96   0.89   1.44     % 16 Skin
        ];
    elseif wavelength == "1064nm"
        % Optical properties of abdominal tissues at 1064 nm
        % opt_prop = [mua, mus, g, n]
        
        opt_prop = [
            0       0       1      1.00;    % 0 Background (Air)
        
            0.11    18.0    0.90   1.40;    % 1 Liver, blood-rich parenchymal organ
            0.075   13.5    0.91   1.39;    % 2 Right kidney
            0.13    20.0    0.91   1.40;    % 3 Spleen, highly blood-rich organ
            0.060   9.5     0.92   1.38;    % 4 Pancreas
        
            0.130   20.0    0.985  1.40;    % 5 Aorta, approximated as blood vessel
            0.115   18.0    0.985  1.37;    % 6 Inferior vena cava, venous blood vessel
        
            0.065   11.0    0.90   1.42;    % 7 Right adrenal gland
            0.065   11.0    0.90   1.42;    % 8 Left adrenal gland
        
            0.020   3.0     0.95   1.35;    % 9 Gallbladder, bile/fluid dominated
            0.055   9.0     0.90   1.38;    % 10 Esophagus
            0.060   10.0    0.91   1.38;    % 11 Stomach
            0.060   10.0    0.91   1.38;    % 12 Duodenum
        
            0.075   13.5    0.91   1.39;    % 13 Left kidney
        
            0.019   14.64   0.89   1.43;    % 14 Bone, skull-like interpolated value at 1064 nm
            0.0054  8.59    0.89   1.45;    % 15 Fat
            0.017   18.45   0.89   1.44     % 16 Skin
        ];
    end


    opt_prop(:,2) = opt_prop(:,2) .* (1 - opt_prop(:,3)); % mu_s
    cfg.prop = opt_prop;
    r = [0,0,0,0;2.*rand(size(opt_prop(2:end,:))) - 1];
    map = zeros(size(opt_prop));
    map(:,1:2) = 0.2;
    cfg.prop = cfg.prop .* r .* map + cfg.prop;
    cfg.unitinmm = 1/8; 
    cfg.issrcfrom0 = 1;

   
    %%
    % Light source setting
    src_per_ring = 8;      
    ring_num = 2;          
    src_num = src_per_ring * ring_num;
    
    cfg.nphoton = 1e8;
    cfg.srctype = 'gaussian';
    cfg.srcpos = zeros(src_num, 3);
    cfg.srcdir = zeros(src_num, 3);
    cfg.srcparam1 = zeros(src_num, 4);
    cfg.srcparam2 = zeros(src_num, 4);
    
    r = min(Nx, Ny) * 0.45;
    center_x = Nx / 2;
    center_y = Ny / 2;
    

    z_positions = [
        depth * 0.35;
        depth * 0.65
    ];
    
    src_idx = 1;
    for ring_id = 1:ring_num
        z_axis = z_positions(ring_id);
    
        for i = 1:src_per_ring
            theta = 2 * pi / src_per_ring * (i - 1);
    
            cfg.srcpos(src_idx, :) = [
                center_x + r * sin(theta), ...
                center_y + r * cos(theta), ...
                z_axis
            ];
    
            cfg.srcdir(src_idx, :) = [
                center_x, center_y, z_axis
            ] - cfg.srcpos(src_idx, :);
    
            cfg.srcdir(src_idx, :) = cfg.srcdir(src_idx, :) / norm(cfg.srcdir(src_idx, :), 2);
    
            cfg.srcparam1(src_idx, :) = [60, 0, 0, 0];
    
            src_idx = src_idx + 1;
        end
    end

    %%

    % GPU setting
    cfg.gpuid = char(gpu_id);
    cfg.autopilot = 1;

    % Time setting
    cfg.tstart = 0;
    cfg.tend = 5e-9;
    cfg.tstep = 5e-9;


    % Simulation parameters
    cfg.debuglevel = 'P';
    cfg.outputtype = 'fluence';
    cfg.isspecular = 0;
    cfg.isreflect = 0;
    cfg.issaveref = 0;
    cfg.lambda = str2double(erase(wavelength, "nm"));

    % get absorption coeffcient
    mua = zeros(size(seg));
    for i = 1:16
        mua(seg == i) = cfg.prop(i + 1, 1);
    end
    [fluence,detpt,vol,seeds,traj] = mcxlab(cfg);
    flux = sum(fluence.data, 4);
    pressure = flux(:, :, 1:depth) .* mua(:, :, end - depth + 1:end);
    log_flux = log10(flux + 1e-12);

    flux = flux(rows_in_padded, cols_in_padded, :);
    log_flux = log_flux(rows_in_padded, cols_in_padded, :);
    pressure = pressure(rows_in_padded, cols_in_padded, :);
    mua = mua(rows_in_padded, cols_in_padded, :);

    info = niftiinfo(seg_path); 
    info.ImageSize = size(seg_processed);
    info.Datatype = 'single'; 
    
    % break

    flux_file = fullfile(flux_path, case_name + "_flux");
    log_flux_file = fullfile(flux_log_path, case_name + "_logFlux");
    press_file = fullfile(press_path, case_name + "_IP");
    mua_file = fullfile(mua_path, case_name + "_mua");
    
    niftiwrite(single(flux), flux_file, info, 'Compressed', true);
    niftiwrite(single(log_flux), log_flux_file, info, 'Compressed', true);
    niftiwrite(single(pressure), press_file, info, 'Compressed', true);
    
    info.Datatype = 'single';
    niftiwrite(single(mua), mua_file, info, 'Compressed', true);
end
end
