/*----------------------
   Copyright (C): OpenGATE Collaboration
This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateParticleFilterMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEPARTFILTERMESSENGER_HH
#define GATEPARTFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithAString.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithoutParameter.hh"

class GateParticleFilter;

class GateParticleFilterMessenger : public  G4UImessenger
{
public:
  GateParticleFilterMessenger(GateParticleFilter* partFilter);
  virtual ~GateParticleFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateParticleFilter * pParticleFilter;

  G4UIcmdWithAString* pAddParticleCmd;
  G4UIcmdWithAnInteger* pAddParticleZCmd;
  G4UIcmdWithAnInteger* pAddParticleACmd;
  G4UIcmdWithAnInteger* pAddParticlePDGCmd;
  G4UIcmdWithAString* pAddParentParticleCmd;
  G4UIcmdWithAString* pAddDirectParentParticleCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEPARTFILTERMESSENGER_HH */
