function get_ip_from_model(model, dataset, input_root, output_root)
%GET_IP_FROM_MODEL Generate absorption and approximate initial-pressure maps.
%
% Input:  <input_root>/<model>/<dataset>
% Output: <output_root>/<model>/<dataset>_p0
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
output_dir  = fullfile(output_root, model, dataset + "_p0");

if ~isfolder(phantom_dir)
    error("Input directory does not exist: %s", phantom_dir);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

file_list = dir(fullfile(phantom_dir, "*.nii.gz"));
len = numel(file_list);

fprintf("Found %d nii.gz files in %s\n", len, phantom_dir);

for f = 1:len
    nii_path = fullfile(file_list(f).folder, file_list(f).name);

    [~, base, ext] = fileparts(nii_path);
    if strcmpi(ext, ".gz")
        [~, name] = fileparts(base);   % base: "name.nii" -> name: "name"
    else
        name = base;
    end

    info = niftiinfo(nii_path);
    tissue_map = single(niftiread(info));

    [Nx, Ny, Nz] = size(tissue_map);
    fprintf('(%d/%d) Loaded %s: %d × %d × %d\n', f, len, name, Nx, Ny, Nz);

    %% mu_a 
    mu_a = zeros(size(tissue_map), 'single');

    mu_a(tissue_map == 0) = 0.002;  % Background 
    mu_a(tissue_map == 1) = 0.014;   % White-Matter
    mu_a(tissue_map == 2) = 0.036;   % Gray-Matter
    mu_a(tissue_map == 3) = 0.004;   % CSF
    mu_a(tissue_map == 4) = 0.020;   % Muscle
    mu_a(tissue_map == 5) = 0.018;   % Scalp
    mu_a(tissue_map == 6) = 0.05;   % Eye balls
    mu_a(tissue_map == 7) = 0.016;   % Compact bone
    mu_a(tissue_map == 8) = 0.016;   % Spongy bone
    mu_a(tissue_map == 9) = 0.35;   % Vessel
    
    mat_path = fullfile(output_dir, name + "_mu.mat");
    p0 = mu_a;
    save(mat_path, "p0", "-v7.3");
    %% fluence
    z_norm = linspace(0, 1, Nz);                 % 0~1
    [~, ~, Z] = ndgrid(1:Nx, 1:Ny, z_norm);      % Z: Nx×Ny×Nz

    delta = 0.60;                                 % penetration depth 
    fluence = exp(-Z / delta);

    %% p0
    Gamma = 1.0;
    p0 = Gamma .* mu_a .* fluence;
    p0 = max(p0, 0);

    p0max = max(p0(:));
    if p0max > 0
        p0 = p0 ./ p0max;
    end

    apply_smoothing = true;
    if apply_smoothing
        sigma = 0.3;
        p0 = imgaussfilt3(p0, sigma);
    end

    %% saving
    mat_path = fullfile(output_dir, name + "_p0.mat");
    save(mat_path, "p0", "-v7.3");
    fprintf("Saved: %s\n", mat_path);

end
end
