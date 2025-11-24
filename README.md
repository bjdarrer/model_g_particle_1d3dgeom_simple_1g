# model_g_particle_1d3dgeom_simple_1g

Model G Particle – 1D / 1D cylindrical / 1D spherical
Clean, no-checkpoint version that just runs and makes an MP4.

Simulating a 1D Model G soliton particle, with options of 1D cartesian, or 1D symmetry in: 2D (cylindrical) and 3D (spherical).

- Written by Brendan Darrer aided by ChatGPT5.1 date: 23rd November 2025 - adapted 24.11.2025 16:07 GMT
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb
- with ChatGPT5.1 writing the code and Brendan guiding it to produce a clean code.

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

Usage examples:
    python3 model_g_particle_1d3dgeom_simple_1g.py.py --geometry 1d
    python3 model_g_particle_1d3dgeom_simple_1g.py --geometry 1d_cyl
    python3 model_g_particle_1d3dgeom_simple_1g.py --geometry 1d_sph

Requires:
    pip install numpy scipy matplotlib imageio imageio[ffmpeg]
