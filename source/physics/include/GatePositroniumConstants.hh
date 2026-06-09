/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About class: Namespace of physical constants for positronium physics: para-Ps lifetime (0.1244 ns), ortho-Ps mean lifetime (142 ns), para-to-ortho fraction (1/3), and hyperfine coefficient (372).
 **/

#ifndef GatePositroniumConstants_hh
#define GatePositroniumConstants_hh

namespace GatePositroniumConstants
{
  constexpr double kParaPsLifetime_ns = 0.1244;
  constexpr double kParaToOrthoPsFraction = 1.0/3.0;
  constexpr double kOrthoPsMeanLifetime_ns = 142.;  
  constexpr double kHyperfineCoefficient = 372;
}

#endif

