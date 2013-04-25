/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateHybridProcess
  \author fabien.baldacci@creatis.insa-lyon.fr
*/

#ifndef GATEHYBRIDPROCESS_HH
#deinfe GATEHYBRIDPROCESS_HH

#include "G4WrapperProcess.hh"

class GateHybridProcess : publis G4WrapperProcess
{
public:
  GateHybridProcess(G4String t);
  virtual ~GateHybridProcess();
  
  virtual G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step);
  
private:
  

}
