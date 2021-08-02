#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2019-2021  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# List all datafiles
package_files = []
package_dir = os.path.join(os.path.join(os.path.dirname(__file__), "libnlif"))
for root, dirs, files in os.walk(os.path.join(package_dir, "cpp")):
    root = os.path.relpath(root, package_dir)
    for f in files:
        package_files.append(os.path.join(root, f))

# Run the actual setup
setup(
    name='libnlif',
    packages=find_packages(),
    package_data={
        "nlif": package_files
    },
    version='1.0',
    author='Andreas Stöckel',
    author_email='astoecke@uwaterloo.ca',
    description='Weight solver and simulator for multi-compartment LIF neurons',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/astoeckel/libnlif',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=[
        "numpy>=1.16.3",
    ],
)

