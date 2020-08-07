/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateComptonCameraActorMessenger
*/

#include "GateConfiguration.h"

#ifndef GATECOMPTONCAMERAACTORMESSENGER_HH
#define GATECOMPTONCAMERAACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

#include "GateActorMessenger.hh"

class GateComptonCameraActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateComptonCameraActorMessenger : public GateActorMessenger
{
public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateComptonCameraActorMessenger(GateComptonCameraActor * v);
  /// Destructor
  virtual ~GateComptonCameraActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateComptonCameraActor * pActor;

  /// Command objects
  G4UIcmdWithABool          * pSaveHitsTree;
  G4UIcmdWithABool          * pSaveSinglesTree;
  G4UIcmdWithABool          * pSaveCoincidencesTree;
  G4UIcmdWithABool          * pSaveCoincidenceChainsTree;
  G4UIcmdWithABool          * pSaveEventInfoTree;


  //Include flags for variables it affects to all the trees (hits, singles, coincidences)
  //(only checked singles variables)
  G4UIcmdWithABool* pEnableEnergyCmd;
  G4UIcmdWithABool* pEnableEnergyIniCmd;
  G4UIcmdWithABool* pEnableEnergyFinCmd;
  G4UIcmdWithABool* pEnableTimeCmd;
  G4UIcmdWithABool* pEnablePositionXCmd;
  G4UIcmdWithABool* pEnablePositionYCmd;
  G4UIcmdWithABool* pEnablePositionZCmd;
  G4UIcmdWithABool* pEnableLocalPositionXCmd;
  G4UIcmdWithABool* pEnableLocalPositionYCmd;
  G4UIcmdWithABool* pEnableLocalPositionZCmd;
  G4UIcmdWithABool* pEnableSourcePositionXCmd;
  G4UIcmdWithABool* pEnableSourcePositionYCmd;
  G4UIcmdWithABool* pEnableSourcePositionZCmd;
  //volume identification
  G4UIcmdWithABool* pEnableVolumeIDCmd;
  //G4UIcmdWithABool* pEnableLayerNameCmd;
  G4UIcmdWithABool* pEnableSourceEnergyCmd;
  G4UIcmdWithABool* pEnableSourcePDGCmd;
  G4UIcmdWithABool* pEnablenCrystalConvCmd;
  G4UIcmdWithABool* pEnablenCrystalComptCmd;
  G4UIcmdWithABool* pEnablenCrystalRaylCmd;




  //
  G4UIcmdWithAString        * pNameOfAbsorberSDVol;
  G4UIcmdWithAString        * pNameOfScattererSDVol;
  G4UIcmdWithAnInteger      * pNumberofDiffScattererLayers;
  G4UIcmdWithAnInteger      * pNumberofTotScattererLayers;


  G4UIcmdWithABool          * pSourceParentIDSpecification;
  G4UIcmdWithAString        * pFileName4SourceParentID;


}; // end class GateComptonCameraActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATECOMPTONCAMERAACTORMESSENGER_HH */
//#endif
