function optical_sim_ixi(seg_root, output_root, wavelength, gpu_id, random_seed)
%OPTICAL_SIM_IXI Generate IXI photon-flux targets with MCX.
%
% optical_sim_ixi(seg_root, output_root, wavelength, gpu_id, random_seed)
% Generated data are stored below <output_root>/<wavelength>. The random
% seed controls the per-run optical-property perturbations.

arguments
    seg_root (1,1) string
    output_root (1,1) string
    wavelength (1,1) string {mustBeMember(wavelength,["700nm"])} = "700nm"
    gpu_id (1,1) string = "1"
    random_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 0
end

if ~isfolder(seg_root)
    error("Input directory does not exist: %s", seg_root);
end
rng(random_seed, "twister");

flux_log_path = fullfile(output_root, wavelength, "brain_log_flux_3d");
flux_path = fullfile(output_root, wavelength, "brain_flux_3d");
press_path = fullfile(output_root, wavelength, "brain_init_press");
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

for index = 1:length(files)
    % get file absolute path
    full_file_path = fullfile(files(index).folder, files(index).name);
    
    % get the name of folder which contains the file(IXI-xxxx)
    [~, parent_folder] = fileparts(files(index).folder);
    
    case_name = string(parent_folder);
    data = niftiread(full_file_path); 
    info = niftiinfo(full_file_path); 
    
    seg = data;
    [Nx, Ny, d] = size(seg);
    depth = d - 1;
    seg_3d = zeros(Nx, Ny, depth + 1);
    seg_3d(:, :, 1:depth) = seg(:,:, end - depth + 1:end);
    cfg.vol = uint8(seg_3d);

    % Optical properties of wavelength 700nm
    % Strict linear interpolation from 670 nm and 810 nm
    % opt_prop = [mua, mus, g, n]
    
    opt_prop = [0        0        1       1.00;    % Background Media
                0.0747   39.65    0.854   1.37;    % White-Matter
                0.0217   8.16     0.898   1.43;    % Gray-Matter
                0.00087  0.0909   0.89    1.33;    % CSF
                0.0484   6.93     0.89    1.40;    % Muscle
                0.0536   21.96    0.89    1.37;    % Scalp / Skin
                0.00087  0.0909   0.89    1.33;    % Eye_balls, approximated as CSF
                0.0187   12.24    0.89    1.43;    % Compact_bone / Skull
                0.0187   12.24    0.89    1.43;    % Spongy_bone / Skull
                12.57   30.75    0.989   1.40];    % Vessel



    % Optical properties of wavelength 1064nm
    % https://pmc.ncbi.nlm.nih.gov/articles/PMC6366475/
    % opt_prop = [mua, mus, g, n]
    % opt_prop = [0        0        1      1;      % Background Media
    %             0.105    30       0.88   1.37;   % White-Matter
    %             0.053    5.9      0.91   1.43;   % Gray-Matter
    %             0.0144   0.091    0.89   1.33;   % CSF
    %             0.056    5.00     0.89   1.40;   % Muscle
    %             0.017    18.45    0.89   1.37;   % Scalp / Skin
    %             0.0144   0.091    0.89   1.33;   % Eye_balls, approximated as CSF
    %             0.019    14.64    0.89   1.43;   % Compact_bone / Skull
    %             0.019    14.64    0.89   1.43;   % Spongy_bone / Skull
    %             13.0    20       0.985  1.40];  % Vessel
    

    opt_prop(:,2) = opt_prop(:,2) .* (1 - opt_prop(:,3));
    cfg.prop = opt_prop;
    r = [0,0,0,0;2.*rand(size(opt_prop(2:end,:))) - 1];
    map = zeros(size(opt_prop));
    map(:,1:2) = 0.2;
    cfg.prop = cfg.prop .* r .* map + cfg.prop;
    cfg.unitinmm = 0.9375/4; % change real spacing
    cfg.issrcfrom0 = 1;

    % Light source setting
    src_num = 9;
    z_axis = 31;
    cfg.nphoton = 1e8;
    cfg.srctype = 'gaussian';
    cfg.srcpos = zeros(src_num,3);
    cfg.srcdir = zeros(src_num,3);
    cfg.srcparam1 = zeros(src_num,4);
    cfg.srcparam2 = zeros(src_num,4);
    % Top Gaussian source
    cfg.srcpos(1, :) = [128, 128, depth + 1];
    cfg.srcdir(1, :) = [0, 0, -1];
    cfg.srcparam1(1, :) = [120, 0, 0, 20];
    % Ring source
    for i = 2:src_num
        cfg.srcpos(i, :) = [128 + 120*sin(2 * pi/(src_num - 1)*(i - 2)), 128 + 120 * cos(2 * pi/(src_num - 1)*(i - 2)), z_axis];
        cfg.srcdir(i, :) = [128, 128, z_axis] - cfg.srcpos(i, :);
        cfg.srcdir(i, :) = cfg.srcdir(i, :)/norm(cfg.srcdir(i, :), 2);
        cfg.srcparam1(i, :) = [60, 0, 0, 0];
    end

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
    cfg.isreflect = 1;
    cfg.issaveref = 1;
    cfg.lambda = str2double(erase(wavelength, "nm"));

    % get absorption coeffcient
    mua = zeros(size(seg));
    for i = 1:9
        mua(seg == i) = cfg.prop(i + 1, 1);
    end
    [fluence,detpt,vol,seeds,traj] = mcxlab(cfg);
    flux = sum(fluence.data, 4);
    pressure = flux(:, :, 1:d) .* mua;
    
    log_flux = log10(flux + 1e-12);
    log_flux = log_flux(:,:,1:d);

    info.ImageSize = size(log_flux);
    info.Datatype = 'single'; 
    % 3. 保存
    flux_file = fullfile(flux_path, case_name + "_flux");
    log_flux_file = fullfile(flux_log_path, case_name + "_logFlux");
    press_file = fullfile(press_path, case_name + "_IP");
    niftiwrite(single(flux), flux_file, info, 'Compressed', true);
    niftiwrite(single(log_flux), log_flux_file, info, 'Compressed', true);
    niftiwrite(single(pressure), press_file, info, 'Compressed', true);
    % break
end
end
