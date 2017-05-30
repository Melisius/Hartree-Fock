[![Build Status](https://travis-ci.org/Melisius/Hartree-Fock.svg?branch=master)](https://travis-ci.org/Melisius/Hartree-Fock)
[![Coverage Status](https://coveralls.io/repos/github/Melisius/Hartree-Fock/badge.svg?branch=master)](https://coveralls.io/github/Melisius/Hartree-Fock?branch=master)

# Hartree-Fock

Hartree-Fock program written in python. See input and setting file, as example files given as inputs.

## Basis sets:

STO-3G  : H, C, N, O

DZ      : H, O

DZP     : H, O

## Properties:

HF energy, MP2 energy

Mulliken charges, ESP fitted charges

Molecular dipole moment, ESP fitted dipoles

Geometry Optimization for HF, analytic gradient

## Schemes:

SCF, DIIS

Overlap integral, Obara-Saika

Electronic kinetic energy integral, Obara-Saika

Multipole integral, Obara-Saika

One electron coulomb integral, MacMurchie-Davidson

ERI, MacMurchie-Davidson


## Requirements:

Numpy

Scipy

