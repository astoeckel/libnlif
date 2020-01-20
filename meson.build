#  libbioneuronqp -- Library solving for synaptic weights
#  Copyright (C) 2020  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

project(
	'libbioneuron',
	['c', 'cpp'],
	default_options: [
		'cpp_std=c++11',
		'c_std=c99',
		'b_ndebug=if-release',
		'buildtype=release',
	])

include_osqp = include_directories('extern/osqp')
include_extern = include_directories('extern')

cc = meson.get_compiler('c')
dep_m = cc.find_library('m', required : false)

dep_eigen3 = dependency('eigen3')
dep_threads = dependency('threads')

lib_osqp = library(
	'osqp',
	[
		'extern/osqp/amd_1.c',
		'extern/osqp/amd_2.c',
		'extern/osqp/amd_aat.c',
		'extern/osqp/amd_defaults.c',
		'extern/osqp/amd_info.c',
		'extern/osqp/amd_order.c',
		'extern/osqp/amd_postorder.c',
		'extern/osqp/amd_post_tree.c',
		'extern/osqp/amd_preprocess.c',
		'extern/osqp/amd_valid.c',
		'extern/osqp/qdldl.c',
		'extern/osqp/qdldl_interface.c',
		'extern/osqp/SuiteSparse_config.c',
		'extern/osqp/auxil.c',
		'extern/osqp/cs.c',
		'extern/osqp/error.c',
		'extern/osqp/kkt.c',
		'extern/osqp/lin_alg.c',
		'extern/osqp/lin_sys.c',
		'extern/osqp/osqp.c',
		'extern/osqp/polish.c',
		'extern/osqp/proj.c',
		'extern/osqp/scaling.c',
		'extern/osqp/util.c',
	],
	include_directories: [include_osqp],
	dependencies: [dep_m],
)

lib_bioneuronqp = library(
	'bioneuronqp',
	[
		'bioneuronqp/bioneuronqp.cpp',
		'bioneuronqp/threadpool.cpp',
	],
	dependencies: [dep_eigen3, dep_threads],
	include_directories: [include_extern],
	link_with: [lib_osqp],
)
