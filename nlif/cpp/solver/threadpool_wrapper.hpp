/*
 *  libnlif -- Multi-compartment LIF simulator and weight solver
 *  Copyright (C) 2017-2021  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef NLIF_THREADPOOL_WRAPPER_H
#define NLIF_THREADPOOL_WRAPPER_H

#include "common.h"
#include "threadpool.hpp"

namespace ThreadpoolWrapper {

static NlifError run(int n_threads, int n_work_items,
               const Threadpool::Kernel &kernel,
               const NlifProgressCallback progress)
{
	// Construct the progress callback
	bool did_cancel = false;
	auto progress_wrapper = [&](size_t cur, size_t max) {
		if (progress) {
			if (!progress(cur, max)) {
				did_cancel = true;
			}
			return !did_cancel;
		}
		return true;
	};

	// Create a threadpool and solve the weights for all neurons. Do not create
	// a threadpool if there is only one set of weights to solve for, or the
	// number of threads has explicitly been set to one.
	n_threads = std::min(n_threads, n_work_items);
	if ((n_threads != 1) && (n_work_items > 1)) {
		Threadpool pool(n_threads);
		pool.run(n_work_items, kernel, progress_wrapper);
	}
	else {
		for (int i = 0; i < n_work_items; i++) {
			kernel(i);
			if (progress) {
				if (!progress(i + 1, n_work_items)) {
					return NL_ERR_CANCEL;
				}
			}
		}
	}
	return did_cancel ? NL_ERR_CANCEL : NL_ERR_OK;
}

}  // namespace ThreadpoolWrapper

#endif /* NLIF_THREADPOOL_WRAPPER_H */
