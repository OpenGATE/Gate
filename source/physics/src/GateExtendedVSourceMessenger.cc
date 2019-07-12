#include "GateExtendedVSourceMessenger.hh"
#include "GateExtendedVSource.hh"

GateExtendedVSourceMessenger::GateExtendedVSourceMessenger( GateExtendedVSource* source ) : GateVSourceMessenger( source )
{
 pSource = source;
 InitCommands();
}

GateExtendedVSourceMessenger::~GateExtendedVSourceMessenger() { DeleteCommands(); }

void GateExtendedVSourceMessenger::InitCommands()
{
 G4String cmdName = "";
 
 cmdName = GetDirectoryName() + "setSeedForRandomGenerator";
 pCmdSeedForRandomGenerator = new G4UIcmdWithAnInteger( cmdName, this );
 pCmdSeedForRandomGenerator->SetGuidance( "A seed for TRandom3 in gamma sources" );
 pCmdSeedForRandomGenerator->SetParameterName( "seedRG", false );
 pCmdSeedForRandomGenerator->SetRange( "seedRG>=0" );
 
 cmdName = GetDirectoryName() + "setPromptGammaEnergy";
 pCmdPromptGammaEnergy = new G4UIcmdWithADoubleAndUnit( cmdName, this );
 pCmdPromptGammaEnergy->SetGuidance( "Energy of prompt gamma" );

 cmdName = GetDirectoryName() + "setLinearPolarizationAngle";
 pCmdLinearPolarization = new G4UIcmdWithADoubleAndUnit( cmdName, this );
 pCmdLinearPolarization->SetGuidance( "Set linear polarization by given angle" );

 cmdName = GetDirectoryName()+"setUseUnpolarizedParticles";
 pCmdUseUnpolarizedParticles = new G4UIcmdWithABool (cmdName, this );
 pCmdUseUnpolarizedParticles->SetGuidance( "Set unpolarized particles generation" );
}

void GateExtendedVSourceMessenger::DeleteCommands()
{
 delete pCmdSeedForRandomGenerator;
 delete pCmdPromptGammaEnergy;
 delete pCmdLinearPolarization;
 delete pCmdUseUnpolarizedParticles;
}

void GateExtendedVSourceMessenger::SetNewValue( G4UIcommand* command, G4String newValue )
{
 if ( command == pCmdSeedForRandomGenerator )
  pSource->SetSeedForRandomGenerator( static_cast<unsigned int>( pCmdSeedForRandomGenerator->GetNewIntValue( newValue ) ) );
 else if ( command == pCmdPromptGammaEnergy )
  pSource->SetPromptGammaEnergy( pCmdPromptGammaEnergy->GetNewDoubleValue( newValue ) );
 else if ( command == pCmdLinearPolarization )
  pSource->SetLinearPolarizationAngle( pCmdLinearPolarization->GetNewDoubleRawValue( newValue ) );
 else if ( command == pCmdUseUnpolarizedParticles )
  pSource->SetUnpolarizedParticlesGenerating( pCmdUseUnpolarizedParticles->GetNewBoolValue( newValue ) );
 else
  GateVSourceMessenger::SetNewValue( command, newValue );
}


