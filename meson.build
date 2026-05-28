project(
  'mrmustard',
  'c', 'cpp', 'cython',
  meson_version: '>= 1.8.0',
  default_options: [
    'buildtype=plain',
    'c_std=c17',
    'cpp_std=c++17',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

if host_machine.system() == 'windows' and cc.get_id() != 'msvc'
    error('MrMustard requires Windows to be build with MSVC.')
endif

# Numpy dependency for Cython
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# Use a compile arg instead of include_directories() so the path is accepted
# even when it lives inside the source tree (e.g. an in-project .venv built
# with build isolation disabled).
np_dep = declare_dependency(compile_args: ['-I' + incdir_numpy])

# OpenMP dependency
omp_dep = dependency('openmp', required: get_option('use_openmp'))

if not omp_dep.found()
  warning('OpenMP not found - building without parallel support')
endif

# Math library linking for C code
m_dep = cc.find_library('m', required: false)
if m_dep.found()
  add_project_link_arguments('-lm', language: 'c')
endif

# Project arguements
add_project_arguments(
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  language: ['c', 'cpp']
)

add_project_arguments(
  cc.get_supported_arguments('-ffast-math'),
  language: 'c'
)

# Install the main Python package
install_subdir('mrmustard', 
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
  exclude_files: ['*.pyc', '*.pyo', '*.so', '*.c', '*.cpp', '*.pyx', '*.pxd']
)

subdir('mrmustard/mathlib/cython_lattice') 

# Build configuration summary
summary({
  'Python': py.full_path(),
  'NumPy include': incdir_numpy,
  'OpenMP': omp_dep.found() ? 'enabled' : 'disabled',
  'Build type': get_option('buildtype'),
  'C compiler': cc.get_id(),
  'C++ compiler': cpp.get_id(),
}, section: 'Configuration', bool_yn: true)