# Traditional simulation pipeline

These MATLAB functions generate photon-flux and initial-pressure data used by
PSPA-DT. All filesystem locations are supplied by the caller; the source files
contain no machine-specific dataset or output paths.

## Dependencies

- MATLAB with NIfTI support (`niftiread`, `niftiinfo`, and `niftiwrite`)
- [MCX / MCXLAB](https://mcx.space/) for optical simulation
- [k-Wave](https://www.k-wave.org/) for acoustic simulation and reconstruction
- FLARE23 helper functions `processed_ct_seg` and `symmetry_pad` must be on the
  MATLAB path when running `optical_sim_flare23`

Add the required code to the MATLAB search path before running the pipeline:

```matlab
addpath("/path/to/PSPA-DT/src/trad_sim/fluence");
addpath("/path/to/PSPA-DT/src/trad_sim/anamony");
addpath("/path/to/mcx/mcxlab");
addpath("/path/to/k-Wave");
```

## Optical simulation

```matlab
optical_sim_ixi("/data/IXI", "/output/IXI", "700nm", "1", 42);
optical_sim_flare23("/data/FLARE23", "/output/FLARE23", "1064nm", "1", 42);
```

The arguments are input root, output root, wavelength, MCX GPU identifier, and
random seed. The wavelength, GPU identifier, and seed are optional. Outputs are
written below `<output_root>/<wavelength>/`.

Each input case is discovered recursively using the filename `seg.nii.gz`.
FLARE23 cases must also contain `ct.nii.gz`.

## Initial pressure and acoustic reconstruction

```matlab
get_ip_from_model("PSPA-DT", "IXI", "/data/predictions", "/output/simulation");
sim_recon_fun("PSPA-DT", "IXI", "/output/simulation", "/output/simulation");
```

`get_ip_from_model` reads `<input_root>/<model>/<dataset>` and writes
`<output_root>/<model>/<dataset>_p0`. `sim_recon_fun` reads the dataset and its
`_p0` directory, then creates `_recon`, `_high`, `_dog`, and `_fft` directories.
If `output_root` is omitted, it defaults to `input_root`.
