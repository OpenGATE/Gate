/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef Legacy_GateParaPositronium_hh
#define Legacy_GateParaPositronium_hh

#include "G4ParticleDefinition.hh"

namespace GateLegacy{

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Generate para-positronium definition and set decay chanel for pPs.
 **/
class GateParaPositronium : public G4ParticleDefinition
{
 private:
  static GateParaPositronium* theInstance;
  GateParaPositronium() {}
  ~GateParaPositronium() {}
 public:
  static GateParaPositronium* Definition();
  static GateParaPositronium* ParaPositroniumDefinition();
  static GateParaPositronium* ParaPositronium();
};
}

#endif
