/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVSourceMessenger.hh"
#include "GateVSource.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
//#include "GateUIcmdWithADoubleWithUnitAndInteger.hh"

//For new activity units
//M Chamberland 19/07/2013
#include "G4UnitsTable.hh"

//----------------------------------------------------------------------------------------
GateVSourceMessenger::GateVSourceMessenger(GateVSource* source)
  : GateMessenger(G4String("source/") + source->GetName()),
    m_source(source)
{ 

//    GateSourceDir = new G4UIdirectory("/gate/source/");
//    GateSourceDir->SetGuidance("GATE source manager control.");

//Added new activity units
//M Chamberland, 19/07/2013
 new G4UnitDefinition("kilobecquerel","kBq","Activity",(1.e3)*becquerel);
 new G4UnitDefinition("megabecquerel","MBq","Activity",(1.e6)*becquerel);
 new G4UnitDefinition("gigabecquerel","GBq","Activity",(1.e9)*becquerel);

 new G4UnitDefinition("millicurie","mCi","Activity",(1.e-3)*curie);
 new G4UnitDefinition("microcurie","muCi","Activity",(1.e-6)*curie);

  G4String cmdName;

  cmdName = GetDirectoryName()+"setActivity";
  ActivityCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  ActivityCmd->SetGuidance("Set source initial activity");
  ActivityCmd->SetParameterName("activity",false);
  ActivityCmd->SetUnitCategory("Activity");
  ActivityCmd->SetRange("activity>=0.0");

  cmdName = GetDirectoryName()+"setStartTime";
  StartTimeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  StartTimeCmd->SetGuidance("Set source start time");
  StartTimeCmd->SetParameterName("time",false);
  StartTimeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName()+"setType";
  TypeCmd = new G4UIcmdWithAString(cmdName,this);
  TypeCmd->SetGuidance("Set source type (backtoback/fastI124/phaseSpace/voxel)");

  cmdName = GetDirectoryName()+"setAccolinearityFlag";
  AccolinearityCmd = new G4UIcmdWithABool(cmdName,this);
  AccolinearityCmd->SetGuidance("Force the GATE source with back to back emission to model the accolinearity");
  AccolinearityCmd->SetGuidance("1. true  to force the accolinearity");
  AccolinearityCmd->SetGuidance("1. false to let the emission without accolinearity");

  cmdName = GetDirectoryName()+"setAccoValue";
  AccoValueCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  AccoValueCmd->SetGuidance("Set accolinearity angle value");
  AccoValueCmd->SetParameterName("AccoValue",false);
  AccoValueCmd->SetUnitCategory("Angle");
  AccoValueCmd->SetRange("AccoValue>=0.0");

  cmdName = GetDirectoryName()+"dump";
  DumpCmd = new G4UIcmdWithAnInteger(cmdName,this);
  DumpCmd->SetParameterName("level", true);
  DumpCmd->SetDefaultValue(0);
  DumpCmd->SetGuidance("List of the source properties");

  cmdName = GetDirectoryName()+"verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set GATE source verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  cmdName = GetDirectoryName()+"setForcedUnstableFlag";
  ForcedUnstableCmd = new G4UIcmdWithABool(cmdName,this);
  ForcedUnstableCmd->SetGuidance("Force the GATE source to be unstable with user defined lifetime");
  ForcedUnstableCmd->SetGuidance("1. true  to force the source to be unstable");
  ForcedUnstableCmd->SetGuidance("1. false to let the source to follow the PDG properties on decays");

  cmdName = GetDirectoryName()+"setForcedLifeTime";
  ForcedLifeTimeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  ForcedLifeTimeCmd->SetGuidance("Set source forced lifetime (tau)");
  ForcedLifeTimeCmd->SetParameterName("forcedLifeTime",false);
  ForcedLifeTimeCmd->SetUnitCategory("Time");
  ForcedLifeTimeCmd->SetRange("forcedLifeTime>0.0");

  cmdName = GetDirectoryName()+"setForcedHalfLife";
  ForcedHalfLifeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  ForcedHalfLifeCmd->SetGuidance("Set source forced halftime (T_1/2)");
  ForcedHalfLifeCmd->SetParameterName("forcedHalfLife",false);
  ForcedHalfLifeCmd->SetUnitCategory("Time");
  ForcedHalfLifeCmd->SetRange("forcedHalfLife>0.0");
  
  cmdName = GetDirectoryName() + "useDefaultHalfLife";
  useDefaultHalfLifeCmd= new G4UIcommand(cmdName,this);
  useDefaultHalfLifeCmd->SetGuidance("Set ion halftime to its default one");
  
  /*
  cmdName = GetDirectoryName()+"setSourceTime";
  BeamTimeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  BeamTimeCmd->SetGuidance("Set the time interval of the source");
  BeamTimeCmd->SetParameterName("Time interval",false);
  BeamTimeCmd->SetUnitCategory("Time");
 
  cmdName = GetDirectoryName()+"setNumberOfParticles";
  NbrOfParticlesCmd = new G4UIcmdWithAnInteger(cmdName,this);
  NbrOfParticlesCmd -> SetGuidance("Set the number of particles produced by the source");
  NbrOfParticlesCmd ->SetParameterName("Number of particles",false);
  
  cmdName = GetDirectoryName()+"setSourceWeight";
  WeightCmd = new G4UIcmdWithADouble(cmdName,this);
  WeightCmd->SetGuidance("Set the weight of the source");
  WeightCmd ->SetParameterName("Weight",false);
  */

  cmdName = GetDirectoryName()+"setIntensity";
  IntensityCmd = new G4UIcmdWithADouble(cmdName,this);
  IntensityCmd->SetGuidance("Set the intensity of the source");
  IntensityCmd ->SetParameterName("Intensity",false);

  /*  cmdName = GetDirectoryName()+"setTimeActivity";
  TimeActivityCmd = new G4UIcmdWithAString(cmdName,this);
  TimeActivityCmd->SetGuidance("Set a filename to read time-activity");
 
  cmdName = GetDirectoryName()+"addSlice";
  TimeParticleSliceCmd = new GateUIcmdWithADoubleWithUnitAndInteger(cmdName,this);
  TimeParticleSliceCmd->SetGuidance("Add a slice to the source");
  TimeParticleSliceCmd->SetParameterName("Time","Time unit","Number of particles",false,false,false);*/
  
  cmdName = GetDirectoryName()+"setMinEnergy";
  setMinEnergycmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  cmdName = GetDirectoryName()+"setEnergyRange";
  setEnergyRangecmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  
  cmdName = GetDirectoryName()+"visualize";
  VisualizeCmd = new G4UIcmdWithAString(cmdName,this);
  VisualizeCmd->SetGuidance("Visualize the source in the geometry");
  VisualizeCmd->SetParameterName("count color size",false);

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateVSourceMessenger::~GateVSourceMessenger()
{
  delete ActivityCmd;
  delete StartTimeCmd;
  delete TypeCmd;
  delete DumpCmd;
  delete VerboseCmd;
  delete ForcedUnstableCmd;
  delete ForcedLifeTimeCmd;
  delete ForcedHalfLifeCmd;
  delete useDefaultHalfLifeCmd;
  delete AccolinearityCmd;
  delete AccoValueCmd;
  //delete BeamTimeCmd;
  //delete NbrOfParticlesCmd;
  // delete WeightCmd;
  delete IntensityCmd;
  //delete TimeActivityCmd;
  //delete TimeParticleSliceCmd;
  delete setMinEnergycmd;
  delete setEnergyRangecmd;
//    delete GateSourceDir;
  delete VisualizeCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateVSourceMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == VerboseCmd ) {
    m_source->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  } else if( command == ActivityCmd ) {
    m_source->SetActivity(ActivityCmd->GetNewDoubleValue(newValue));
  } else if( command == StartTimeCmd ) {
    m_source->SetStartTime(StartTimeCmd->GetNewDoubleValue(newValue));
  } else if( command == TypeCmd ) {
    m_source->SetType(newValue);
  } else if( command == DumpCmd ) {
    m_source->Dump(DumpCmd->GetNewIntValue(newValue));
  } else if( command == AccolinearityCmd ) {
    m_source->SetAccolinearityFlag(AccolinearityCmd->GetNewBoolValue(newValue));
  } else if( command == AccoValueCmd ) {
    m_source->SetAccoValue(AccoValueCmd->GetNewDoubleValue(newValue));
  } else if( command == ForcedUnstableCmd ) {
    m_source->SetForcedUnstableFlag(ForcedUnstableCmd->GetNewBoolValue(newValue));
  } else if( command == ForcedHalfLifeCmd ) {
    m_source->SetForcedHalfLife(ForcedHalfLifeCmd->GetNewDoubleValue(newValue));
  } else if (command == useDefaultHalfLifeCmd ) {
     m_source->SetIonDefaultHalfLife();
  }/* else if( command == BeamTimeCmd ) {
     m_source->SetTimeInterval(BeamTimeCmd->GetNewDoubleValue(newValue));
  } else if( command == NbrOfParticlesCmd ) {
    m_source->SetNumberOfParticles(NbrOfParticlesCmd->GetNewIntValue(newValue));
  } else if( command == WeightCmd ) {
    m_source->SetSourceWeight(WeightCmd->GetNewDoubleValue(newValue));
  } else if ( command == TimeActivityCmd ) {
    m_source->SetTimeActivityFilename(newValue);
    }*/ else if ( command == IntensityCmd ) {
    m_source->SetIntensity(IntensityCmd->GetNewDoubleValue(newValue));
  } /*else if ( command == TimeParticleSliceCmd ) {
      double par1;
      char par2[30];
      int par3;
      std::istringstream is(newValue);
      is >> par1 >> par2 >> par3;
      std::ostringstream par;
      par<<par1;
      G4String val(par.str());
      val.append(" ");
      val.append(par2);
      par1 = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(val);
      m_source->AddTimeSlices(par1,par3);
  }*/
  else if( command == setMinEnergycmd ) {
      m_source->GetEneDist()->SetMinEnergy(setMinEnergycmd->GetNewDoubleValue(newValue));
  } else if( command == VisualizeCmd ) {
    m_source->Visualize(newValue);
  }
  else if( command == setEnergyRangecmd ) {
      m_source->GetEneDist()->SetEnergyRange(setEnergyRangecmd->GetNewDoubleValue(newValue));
  }
}
//----------------------------------------------------------------------------------------


