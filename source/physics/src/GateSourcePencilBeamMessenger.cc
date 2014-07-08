/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCEPENCILBEAMMESSENGER_CC
#define GATESOURCEPENCILBEAMMESSENGER_CC

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateSourcePencilBeamMessenger.hh"
#include "GateSourcePencilBeam.hh"
#include "GateUIcmdWithTwoDouble.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UnitsTable.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"


//----------------------------------------------------------------------------------------
  GateSourcePencilBeamMessenger::GateSourcePencilBeamMessenger(GateSourcePencilBeam* source)
: GateVSourceMessenger(source)
{ 
  pSourcePencilBeam = source;
  G4String cmdName;

  //We build a table of emmitance units
  new G4UnitDefinition(    "milimeter*miliradian","mm*mrad"      ,"Emittance",millimeter*milliradian);

  //Particle Type
  cmdName = GetDirectoryName()+"setParticleType";
  pParticleTypeCmd = new G4UIcmdWithAString(cmdName,this);
  //Particle Properties If GenericIon
  cmdName = GetDirectoryName()+"setIonProperties";
  pIonCmd = new G4UIcommand(cmdName,this);
  pIonCmd->SetGuidance("Set properties of ion to be generated:  Z:(int) AtomicNumber, A:(int) AtomicMass, Q:(int) Charge of Ion (in unit of e), E:(double) Excitation energy (in keV).");
  G4UIparameter* param;
  param = new G4UIparameter("Z",'i',false);
  param->SetDefaultValue("1");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("A",'i',false);
  param->SetDefaultValue("1");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("Q",'i',true);
  param->SetDefaultValue("0");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("E",'d',true);
  param->SetDefaultValue("0.0");
  pIonCmd->SetParameter(param);

  //Energy
  cmdName = GetDirectoryName()+"setEnergy";
  pEnergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setSigmaEnergy";
  pSigmaEnergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  //Position
  cmdName = GetDirectoryName()+"setPosition";
  pPositionCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setSigmaX";
  pSigmaXCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setSigmaY";
  pSigmaYCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  //Direction
  cmdName = GetDirectoryName()+"setSigmaTheta";
  pSigmaThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setSigmaPhi";
  pSigmaPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setRotationAxis";
  pRotationAxisCmd = new G4UIcmdWith3Vector(cmdName,this);
  cmdName = GetDirectoryName()+"setRotationAngle";
  pRotationAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);  
  //Correlation Position/Direction
  cmdName = GetDirectoryName()+"setEllipseXThetaEmittance";
  pEllipseXThetaAreaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setEllipseYPhiEmittance";
  pEllipseYPhiAreaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setEllipseXThetaRotationNorm";
  pEllipseXThetaRotationNormCmd = new G4UIcmdWithAString(cmdName,this);
  cmdName = GetDirectoryName()+"setEllipseYPhiRotationNorm";
  pEllipseYPhiRotationNormCmd = new G4UIcmdWithAString(cmdName,this);
  //Configuration of tests
  cmdName = GetDirectoryName()+"setTestFlag";
  pTestCmd = new G4UIcmdWithABool(cmdName,this);
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourcePencilBeamMessenger::~GateSourcePencilBeamMessenger()
{
  //Particle Type
  delete pParticleTypeCmd;
  //Particle Properties If GenericIon
  delete pIonCmd;
  //Energy
  delete pEnergyCmd;
  delete pSigmaEnergyCmd;
  //Position
  delete pPositionCmd;
  delete pSigmaXCmd;
  delete pSigmaYCmd;
  //Direction
  delete pSigmaThetaCmd;
  delete pSigmaPhiCmd;
  delete pRotationAxisCmd;
  delete pRotationAngleCmd;
  //Correlation Position/Direction
  delete pEllipseXThetaAreaCmd;
  delete pEllipseYPhiAreaCmd;
  delete pEllipseXThetaRotationNormCmd;
  delete pEllipseYPhiRotationNormCmd;
  //Configuration of tests
  delete pTestCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourcePencilBeamMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  //Particle Type
  if (command == pParticleTypeCmd) {pSourcePencilBeam->SetParticleType(newValue);  }
  //Particle Properties If GenericIon
  if (command == pIonCmd) {pSourcePencilBeam->SetIonParameter(newValue);  }
  //Energy
  if (command == pEnergyCmd) {pSourcePencilBeam->SetEnergy(pEnergyCmd->GetNewDoubleValue(newValue));  }
  if (command == pSigmaEnergyCmd) {pSourcePencilBeam->SetSigmaEnergy(pSigmaEnergyCmd->GetNewDoubleValue(newValue));  }
  //Position
  if (command == pPositionCmd)   pSourcePencilBeam->SetPosition(pPositionCmd->GetNew3VectorValue(newValue));
  if (command == pSigmaXCmd) {pSourcePencilBeam->SetSigmaX(pSigmaXCmd->GetNewDoubleValue(newValue));  }
  if (command == pSigmaYCmd) {pSourcePencilBeam->SetSigmaY(pSigmaYCmd->GetNewDoubleValue(newValue));  }
  //Direction
  if (command == pSigmaThetaCmd) {pSourcePencilBeam->SetSigmaTheta(pSigmaThetaCmd->GetNewDoubleValue(newValue));  }
  if (command == pSigmaPhiCmd) {pSourcePencilBeam->SetSigmaPhi(pSigmaPhiCmd->GetNewDoubleValue(newValue));  }
  if (command == pRotationAxisCmd) {pSourcePencilBeam->SetRotationAxis(pRotationAxisCmd->GetNew3VectorValue(newValue)); }
  if (command == pRotationAngleCmd) {pSourcePencilBeam->SetRotationAngle(pRotationAngleCmd->GetNewDoubleValue(newValue));  }
  //Correlation Position/Direction
  if (command == pEllipseXThetaAreaCmd) {pSourcePencilBeam->SetEllipseXThetaArea(pEllipseXThetaAreaCmd->GetNewDoubleValue(newValue));}
  if (command == pEllipseYPhiAreaCmd) {pSourcePencilBeam->SetEllipseYPhiArea(pEllipseYPhiAreaCmd->GetNewDoubleValue(newValue));}
  if (command == pEllipseXThetaRotationNormCmd) {pSourcePencilBeam->SetEllipseXThetaRotationNorm(newValue);  }
  if (command == pEllipseYPhiRotationNormCmd) {pSourcePencilBeam->SetEllipseYPhiRotationNorm(newValue);  }
  //Tests
  if (command == pTestCmd) {pSourcePencilBeam->SetTestFlag(pTestCmd->GetNewBoolValue(newValue)); }
}
#endif

#endif
