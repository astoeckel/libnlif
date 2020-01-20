#  libbioneuron -- Library solving for synaptic weights
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
	default_options: ['cpp_std=c++11', 'b_ndebug=if-release'])

include_osqp = include_directories('lib/osqp')
include_lib = include_directories('lib')

cc = meson.get_compiler('c')
dep_m = cc.find_library('m', required : false)

dep_eigen3 = dependency('eigen3')
dep_threads = dependency('threads')

lib_osqp = library(
	'osqp',
	[
		'lib/osqp/amd_1.c',
		'lib/osqp/amd_2.c',
		'lib/osqp/amd_aat.c',
		'lib/osqp/amd_defaults.c',
		'lib/osqp/amd_info.c',
		'lib/osqp/amd_order.c',
		'lib/osqp/amd_postorder.c',
		'lib/osqp/amd_post_tree.c',
		'lib/osqp/amd_preprocess.c',
		'lib/osqp/amd_valid.c',
		'lib/osqp/qdldl.c',
		'lib/osqp/qdldl_interface.c',
		'lib/osqp/SuiteSparse_config.c',
		'lib/osqp/auxil.c',
		'lib/osqp/cs.c',
		'lib/osqp/error.c',
		'lib/osqp/kkt.c',
		'lib/osqp/lin_alg.c',
		'lib/osqp/lin_sys.c',
		'lib/osqp/osqp.c',
		'lib/osqp/polish.c',
		'lib/osqp/proj.c',
		'lib/osqp/scaling.c',
		'lib/osqp/util.c',
	],
	include_directories: [include_osqp],
	dependencies: [dep_m],
)

lib_bioneuron = library(
	'bioneuron',
	[
		'bioneuron/bioneuron.cpp',
		'bioneuron/threadpool.cpp',
	],
	dependencies: [dep_eigen3, dep_threads],
	include_directories: [include_lib],
	link_with: [lib_osqp],
)
