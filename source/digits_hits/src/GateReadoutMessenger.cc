/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GateReadoutMessenger
    \brief  Messenger for the GateReadout

    \sa GateReadout, GateReadoutMessenger
*/


#include "GateReadoutMessenger.hh"
#include "GateReadout.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"


GateReadoutMessenger::GateReadoutMessenger (GateReadout* itsReadout)
:GateClockDependentMessenger(itsReadout), m_Readout(itsReadout)
{
	//	G4cout<<"GateReadoutMessenger::GateReadoutMessenger"<<G4endl;
    G4String cmdName;

    // Command for choosing the depth of application of the Readout
    cmdName = GetDirectoryName()+"setDepth";
    SetDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetDepthCmd->SetGuidance("Defines the 'depth' of the Readout:");
    SetDepthCmd->SetGuidance("pulses will be summed up if their volume IDs are identical up to this depth.");
    SetDepthCmd->SetGuidance("For instance, the default depth is 1: ");
    SetDepthCmd->SetGuidance("this means that pulses will be considered as taking place in a same block ");
    SetDepthCmd->SetGuidance("if their volume IDs are identical up to a depth of 1, i.e. the first two figures (depth 0 + depth 1)");

    // Command for choosing the policy to create the final pulse (S. Stute)
    // Note: in gate releases until v7.0 included, there was only one policy which was something like "TakeTheWinnerInEnergyForFinalPosition"
    //       we now introduce an option to choose a PMT like Readout by using a policy like "TakeTheCentroidInEnergyForFinalPosition".
    //       So we add a command to choose between the two policy.
    cmdName = GetDirectoryName()+"setPolicy";
    SetPolicyCmd = new G4UIcmdWithAString(cmdName,this);
    SetPolicyCmd->SetGuidance("Defines the policy to be used to compute the final pulse position.");
    SetPolicyCmd->SetGuidance("There are three options: 'TakeWinner', 'TakeEnergyCentroid1' and 'TakeEnergyCentroid2'");
    SetPolicyCmd->SetGuidance("  --> 'TakeEnergyWinner': the final position will be the one for which the maximum energy was deposited");
    SetPolicyCmd->SetGuidance("  --> 'TakeEnergyCentroid': the energy centroid is computed based on crystal indices, meaning that the 'crystal' component of the system must be used. The depth is thus ignored.");
    SetPolicyCmd->SetGuidance("Note: when using the energyCentroid policy, the mother volume of the crystal level MUST NOT have any other daughter volumes declared BEFORE the crystal. Declaring it after is not a problem though.");

    //OK: Choosing the name of the volume where Readout is applied
    cmdName = GetDirectoryName() + "setReadoutVolume";
    SetVolNameCmd = new G4UIcmdWithAString(cmdName,this);
    SetVolNameCmd->SetGuidance("Choose a volume (depth) for Readout (e.g. crystal/module)");

    //OK: Force the name of the volume where Readout is applied for EnergyCentroid policy
    cmdName = GetDirectoryName() + "forceReadoutVolumeForEnergyCentroid";
    ForceDepthCentroidCmd = new G4UIcmdWithABool(cmdName,this);
    ForceDepthCentroidCmd->SetGuidance("Force Depth For Energy Centroid:set to true to activate /setReadoutVolume and /setDepth even if the policy is EnergyCentroid."
    		"(NB:  If the energy centroid policy is used the depth is forced to be at the level just above the crystal level, whatever the system used.)");


     //to add these options later
    /*cmdName = GetDirectoryName() + "setResultingXY";
    SetResultingXYCmd = new G4UIcmdWithAString(cmdName,this);
    SetResultingXYCmd->SetGuidance("Choose the resulting policy for the local position: crystalCenter/exactPostion");

    cmdName = GetDirectoryName() + "setResultingZ";
    SetResultingZCmd = new G4UIcmdWithAString(cmdName,this);
    SetResultingZCmd->SetGuidance("Choose the resulting policy for the local position: crystalCenter/exactPostion/meanFreePath");
*/

}


GateReadoutMessenger::~GateReadoutMessenger()
{
	  delete SetDepthCmd;
	  delete SetPolicyCmd;
	  delete SetVolNameCmd;
	  delete ForceDepthCentroidCmd;

	 // delete SetResultingXYCmd;
	 // delete SetResultingZCmd;
}


void GateReadoutMessenger::SetNewValue(G4UIcommand * aCommand,G4String aString)
{
	//G4cout<<"GateReadoutMessenger::SetNewValue"<<G4endl;
	if( aCommand==SetDepthCmd )
	    { GetReadout()->SetDepth(SetDepthCmd->GetNewIntValue(aString)); }
	  else if ( aCommand==SetVolNameCmd )
	    { GetReadout()->SetVolumeName(aString); }
	  else if ( aCommand==ForceDepthCentroidCmd )
	     { GetReadout()->ForceDepthCentroid(aString); }
	  else if ( aCommand==SetPolicyCmd )
	    { GetReadout()->SetPolicy(aString); }
	  //else if ( aCommand==SetResultingXYCmd )
	   //  { GetReadout()->SetResultingXY(aString); }
	  //else if ( aCommand==SetResultingZCmd )
	  //   { GetReadout()->SetResultingZ(aString); }
	  else
		  GateClockDependentMessenger::SetNewValue(aCommand,aString);
}













