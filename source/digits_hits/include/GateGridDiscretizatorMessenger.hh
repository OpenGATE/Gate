/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*
  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GridDiscretizator.cc for more detals
  // OK GND 2022
  
  \class  GridDiscretizatorMessenger
  \brief  Messenger for the GridDiscretizator
  \sa GridDiscretizator, GridDiscretizatorMessenger
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/


#ifndef GateGridDiscretizatorMessenger_h
#define GateGridDiscretizatorMessenger_h 1

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
class GateGridDiscretizator;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;

class GateGridDiscretizatorMessenger : public GateClockDependentMessenger
{
  public:
  
    GateGridDiscretizatorMessenger(GateGridDiscretizator*);
    virtual ~GateGridDiscretizatorMessenger();
  
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline  void SetNewValue2(G4UIcommand* aCommand, G4String aString);


private:

  std::vector<G4UIdirectory*> m_volDirectory;
  GateGridDiscretizator* m_GateGridDiscretizator;
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








