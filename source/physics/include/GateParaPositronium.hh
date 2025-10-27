/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateParaPositronium_hh
#define GateParaPositronium_hh

#include "G4ParticleDefinition.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  About class: Generate para-positronium definition and set decay chanel for pPs.
 **/
class GateParaPositronium : public G4ParticleDefinition
{
 public:
  static GateParaPositronium* Definition();
  static GateParaPositronium* ParaPositroniumDefinition();
  static GateParaPositronium* ParaPositronium();

 private:
  static GateParaPositronium* theInstance;

  GateParaPositronium() = default; 
  ~GateParaPositronium() override = default; 

 public:
  GateParaPositronium(const GateParaPositronium&) = delete;
  GateParaPositronium& operator=(const GateParaPositronium&) = delete;

};

#endif
