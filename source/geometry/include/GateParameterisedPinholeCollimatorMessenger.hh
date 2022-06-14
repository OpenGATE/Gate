/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateParameterisedPinholeCollimatorMessenger_h
#define GateParameterisedPinholeCollimatorMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateVisAttributesMessenger;
class GateParameterisedPinholeCollimator;
class G4UIdirectory;

class GateParameterisedPinholeCollimatorMessenger: public GateVolumeMessenger
{
public:
  GateParameterisedPinholeCollimatorMessenger(GateParameterisedPinholeCollimator* itsInserter);
  ~GateParameterisedPinholeCollimatorMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  virtual inline GateParameterisedPinholeCollimator* GetCollimatorInserter()
  { return (GateParameterisedPinholeCollimator*) GetVolumeCreator(); }

private:

  G4String                    name_Geometry;
  G4UIdirectory*              dir_Geometry;


  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionX1Cmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionY1Cmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionX2Cmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionY2Cmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorHeightCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorRotRadiusCmd;
  G4UIcmdWithAString*  CollimatorInpitFileCmd;


  GateVisAttributesMessenger* visAttributesMessenger;
};

#endif
