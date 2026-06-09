/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef Legacy_GateOrthoPositronium_hh
#define Legacy_GateOrthoPositronium_hh

#include "G4ParticleDefinition.hh"

namespace GateLegacy{

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Generate ortho-positronium definition and set decay chanel for oPs.
 **/
class GateOrthoPositronium : public G4ParticleDefinition
{
 private:
  static GateOrthoPositronium* theInstance;
  GateOrthoPositronium() {}
  ~GateOrthoPositronium() {}
 public:
  static GateOrthoPositronium* Definition();
  static GateOrthoPositronium* OrthoPositroniumDefinition();
  static GateOrthoPositronium* OrthoPositronium();
};
}

#endif
