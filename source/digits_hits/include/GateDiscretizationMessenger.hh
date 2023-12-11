/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*
  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretization.cc for more detals
  // OK GND 2022
  
  \class  GateDiscretizationMessenger
  \brief  Messenger for the GateDiscretization
  \sa GateDiscretization, GateDiscretizationMessenger
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/


#ifndef GateDiscretizationMessenger_h
#define GateDiscretizationMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"


#include <vector>

class G4UIdirectory;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class GateDiscretization;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;

class GateDiscretizationMessenger : public GateClockDependentMessenger
{
  public:
  
    GateDiscretizationMessenger(GateDiscretization*);
    virtual ~GateDiscretizationMessenger();
  
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline  void SetNewValue2(G4UIcommand* aCommand, G4String aString);


private:

  std::vector<G4UIdirectory*> m_volDirectory;
  GateDiscretization* m_GateDiscretization;
  G4UIcmdWithAString          *DiscCmd;
  G4UIcmdWithADoubleAndUnit*   pStripOffsetX;
  G4UIcmdWithADoubleAndUnit*   pStripOffsetY;
  G4UIcmdWithADoubleAndUnit*   pStripOffsetZ;

  G4UIcmdWithADoubleAndUnit*   pStripWidthX;
  G4UIcmdWithADoubleAndUnit*   pStripWidthY;
  G4UIcmdWithADoubleAndUnit*   pStripWidthZ;

  G4UIcmdWithAnInteger*    pNumberStripsX;
  G4UIcmdWithAnInteger*     pNumberStripsY;
  G4UIcmdWithAnInteger*     pNumberStripsZ;

  G4UIcmdWithAnInteger*    pNumberReadOutBlocksX;
  G4UIcmdWithAnInteger*     pNumberReadOutBlocksY;
  G4UIcmdWithAnInteger*     pNumberReadOutBlocksZ;

  G4String m_name;
  G4int m_count;

};

#endif








