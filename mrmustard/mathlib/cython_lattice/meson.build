# Static C++ library for lattice strategies methods
lattice_strategies_lib = static_library(
    'lattice_strategies_lib',
    'Strategies.cpp',
    include_directories: inc_np,
    dependencies: omp_dep,
)

# Create a shared dependency list to reduce duplication
cython_ext_deps = [py_dep, omp_dep]

py.extension_module(
  'vanilla',
  'vanilla.pyx',
  dependencies: cython_ext_deps,
  include_directories: inc_np,
  install: true,
  subdir: 'mrmustard/mathlib/cython_lattice'
)
py.extension_module(
  'strategies',
  'strategies.pyx',
  link_with : [lattice_strategies_lib],
  dependencies: cython_ext_deps,
  include_directories: inc_np,
  install: true,
  subdir: 'mrmustard/mathlib/cython_lattice',
  override_options: ['cython_language=cpp'],
)
py.extension_module(
  'utils',
  'utils.pyx',
  dependencies: cython_ext_deps,
  include_directories: inc_np,
  install: true,
  subdir: 'mrmustard/mathlib/cython_lattice'
)
