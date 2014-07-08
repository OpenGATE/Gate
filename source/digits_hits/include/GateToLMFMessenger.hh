/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \class GateToLMFMessenger
    \brief  This class is the Messenger of GateToLMF

    - GateToLMFMessenger  - by luc.simon@iphe.unil.ch (12 Avr 2002)


    - It allows to script gate/output/lmf/setLMFFileName/.
   \sa GateToLMF
   \sa GateOutputModuleMessenger
*/

#ifndef GateToLMFMessenger_h
#define GateToLMFMessenger_h 1

#include "GateConfiguration.h"
#ifdef GATE_USE_LMF

#include "GateOutputModuleMessenger.hh"

class GateToLMF;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToLMFMessenger: public GateOutputModuleMessenger
{
public:
  GateToLMFMessenger(GateToLMF* gateToLMF);//!< Constructor.
  ~GateToLMFMessenger();//!< Destructor.
/*!
If the command is not known in this class, we pass it to the "mother clas". If the command is known we execute it there.
\sa G4UImessenger
\sa GATEToASCIIMessenger
\sa GATEOutputModuleMessenger
 */
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateToLMF*            m_gateToLMF;
  G4UIcmdWithAString*   GetFileNameCmd;
  G4UIcmdWithAString*   SetInputDataCmd;  //!< The UI command "set input data name"




  /*!
    Command for the LMF headers filling
  */

  G4UIcmdWithABool*     GetCoincidenceBoolCmd;
  G4UIcmdWithABool*     GetDetectorIDBoolCmd;
  G4UIcmdWithABool*     GetEnergyBoolCmd;
  G4UIcmdWithABool*     GetNeighbourBoolCmd;
  G4UIcmdWithAnInteger* GetNeighbourhoodOrderCmd;
  G4UIcmdWithABool*     GetGantryAxialPosBoolCmd;
  G4UIcmdWithABool*     GetGantryAngularPosBoolCmd;
  G4UIcmdWithABool*     GetSourcePosBoolCmd;
  G4UIcmdWithABool*     GetGateDigiBoolCmd;
  G4UIcmdWithABool*     GetComptonBoolCmd;
  G4UIcmdWithABool*     GetComptonDetectorBoolCmd;
  G4UIcmdWithABool*     GetSourceIDBoolCmd;
  G4UIcmdWithABool*     GetSourceXYZPosBoolCmd;
  G4UIcmdWithABool*     GetGlobalXYZPosBoolCmd;
  G4UIcmdWithABool*     GetEventIDBoolCmd;
  G4UIcmdWithABool*     GetRunIDBoolCmd;



};

#endif
#endif
