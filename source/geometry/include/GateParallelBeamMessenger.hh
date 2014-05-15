/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateParallelBeamMessenger_h
#define GateParallelBeamMessenger_h 1

#include "globals.hh"

#include "GateMessenger.hh"

class GateParallelBeam;

class GateParallelBeamMessenger: public GateMessenger
{
  public:
    GateParallelBeamMessenger(GateParallelBeam* itsInserter);
   ~GateParallelBeamMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateParallelBeam* GetParallelBeamInserter() 
      { return m_inserter; }

  private:
    G4UIcmdWithADoubleAndUnit* ParallelBeamDimensionXCmd;
    G4UIcmdWithADoubleAndUnit* ParallelBeamDimensionYCmd;
    G4UIcmdWithADoubleAndUnit* ParallelBeamHeightCmd;
    G4UIcmdWithADoubleAndUnit* ParallelBeamSeptalThicknessCmd;
    G4UIcmdWithADoubleAndUnit* ParallelBeamInnerRadiusCmd;
    G4UIcmdWithAString*        ParallelBeamMaterialCmd;

    GateParallelBeam*  m_inserter;

};

#endif

