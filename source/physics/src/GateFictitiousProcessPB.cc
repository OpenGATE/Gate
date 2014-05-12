/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateFictitiousProcessPB.hh"
#include "G4FastSimulationManagerProcess.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateFictitiousProcessPB::GateFictitiousProcessPB():GateVProcess("Fictitious")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Fictitious interactions for gammas");
  pMessenger = new GateEMStandardProcessMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateFictitiousProcessPB::CreateProcess(G4ParticleDefinition *)
{
  G4int m_nNumSubProc = 2;
  GateTotalDiscreteProcess* totalFictitiousProcess=new GateTotalDiscreteProcess ( "TotalDiscreteProcess",fUserDefined,m_nNumSubProc,G4Gamma::Definition(),1*keV,5*MeV,10000 );
  totalFictitiousProcess->AddDiscreteProcess (new G4PhotoElectricEffect());
  totalFictitiousProcess->AddDiscreteProcess (new G4ComptonScattering());
//  return new G4ComptonScattering(GetG4ProcessName());
  return totalFictitiousProcess;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFictitiousProcessPB::ConstructProcess(G4ProcessManager * manager)
{
  // If fictitious interaction should be used, the corresponding FastSimulationProcess must be inserted
  G4FastSimulationManagerProcess* m_fastSimulation = new G4FastSimulationManagerProcess ( "FastSimulationManagerProcess4Fictitious" );
  manager->AddDiscreteProcess( m_fastSimulation,0 );
  manager->AddDiscreteProcess(GetProcess());
  GatePETVRTSettings* settings=GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings();
  settings->SetApproximations ( GatePETVRT::kVolumeTrace );
  G4double fictitiousEnergy=settings->GetFictitiousEnergy();
  GateFictitiousFastSimulationModel* model=new GateFictitiousFastSimulationModel (fictitiousEnergy,4.99*MeV );
  settings->RegisterFictitiousFastSimulationModel ( model,true ); // true == model will be deleted with settings
  settings->RegisterTotalDiscreteProcess ( ((GateTotalDiscreteProcess*)GetProcess()),false );
  if ( !model->SetParameters ( settings ) )
  {
    G4cout << "WARNING! FictitiousFastSimulationModel set but not all other commands applied (this might be the case if physics commands are used before volume definition)." << G4endl;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateFictitiousProcessPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateFictitiousProcessPB)
