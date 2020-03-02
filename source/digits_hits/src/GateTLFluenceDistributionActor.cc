/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateTLFluenceDistributionActor.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateTLFluenceDistributionActor::GateTLFluenceDistributionActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateTLFluenceDistributionActor() -- begin\n");

   mEnergyMin = 0;
   mEnergyMax = 1;
   mThetaMin = 0;
   mThetaMax = 180;
   mPhiMin = -180;
   mPhiMax = 180;
  
   mEnergyBins = 50; 
   mThetaBins = 36; 
   mPhiBins = 72;
   
   mEnergyEnabled = false;
   mThetaEnabled = true;
   mPhiEnabled = false;
   mAsciiFileEnabled = false;


  pMessenger = new GateTLFluenceDistributionActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateTLFluenceDistributionActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTLFluenceDistributionActor::~GateTLFluenceDistributionActor()
{
  GateDebugMessageInc("Actor",4,"~GateTLFluenceDistributionActor() -- begin\n");

      // Free resources
    if (mEnergyEnabled)
    {
	  delete pHistEnergy;
	  
	  if (mThetaEnabled)
	    delete pHistEnergyTheta; 

	  if (mPhiEnabled)
	    delete pHistEnergyPhi;	
    }
    if (mThetaEnabled)
    {
	  delete pHistTheta;
	  if (mPhiEnabled)
	    delete pHistThetaPhi; 
    }
    if (mPhiEnabled)
	  delete pHistPhi;
    
    if (mAsciiFileEnabled) 
      if (mAsciiFile.is_open())
	mAsciiFile.close();
    
    delete pTfile;
    
    delete pMessenger;


  GateDebugMessageDec("Actor",4,"~GateTLFluenceDistributionActor() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateTLFluenceDistributionActor::Construct()
{
  GateVActor::Construct();
  
  if (!mEnergyEnabled && !mThetaEnabled && !mPhiEnabled)
  {
    GateError("The GateTLFluenceDistributionActor " << GetObjectName()
              << " does not have any output enabled ...\n Please select at least one ('enableEnergy true' for example)");
    return;
  }

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnablePostUserTrackingAction(false);

  pTfile = new TFile(mSaveFilename,"RECREATE");
  
  // Create the enabled histograms
  if (mEnergyEnabled)
  {
	pHistEnergy = new TH1D("energy","Energy",mEnergyBins,mEnergyMin,mEnergyMax);
	pHistEnergy->SetXTitle("Energy [MeV]");
	
	if (mThetaEnabled){
	  pHistEnergyTheta = new TH2D("energyTheta", "Energy-Theta", mEnergyBins, mEnergyMin, mEnergyMax, mThetaBins, mThetaMin, mThetaMax);
	  pHistEnergyTheta->SetXTitle("Energy [MeV]");
	  pHistEnergyTheta->SetYTitle("Theta [deg]"); 
	}
	
	if (mPhiEnabled){
	  pHistEnergyPhi = new TH2D("energyPhi", "Energy-Phi", mEnergyBins, mEnergyMin, mEnergyMax, mPhiBins, mPhiMin, mPhiMax);
	  pHistEnergyPhi->SetXTitle("Energy [MeV]");
	  pHistEnergyPhi->SetYTitle("Phi [deg]"); 
	}	
  }
  if (mThetaEnabled)
  {
	pHistTheta =  new TH1D("theta","Theta",mThetaBins,mThetaMin,mThetaMax);
	pHistTheta->SetXTitle("Theta [deg]");
	
	if (mPhiEnabled){
	  pHistThetaPhi = new TH2D("thetaPhi", "Theta-Phi", mThetaBins, mThetaMin, mThetaMax, mPhiBins, mPhiMin, mPhiMax);
	  pHistThetaPhi->SetXTitle("Theta [deg]");
	  pHistThetaPhi->SetYTitle("Phi [deg]"); 
	}
  }
  if (mPhiEnabled)
  {
	pHistPhi =  new TH1D("phi","Phi",mPhiBins,mPhiMin,mPhiMax);
	pHistPhi->SetXTitle("Phi [deg]");	
  }
  
  detectorVolume = GetVolume()->GetLogicalVolume()->GetSolid()->GetCubicVolume();
  
  if (mAsciiFileEnabled)
    mAsciiFile.open(mAsciiFileName.data());

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTLFluenceDistributionActor::SaveData()
{
  GateVActor::SaveData();
  pTfile->Write();
  mAsciiFile.flush();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLFluenceDistributionActor::ResetData()
{
  if (mEnergyEnabled)
    pHistEnergy->Reset();
  if (mThetaEnabled)
    pHistTheta->Reset();
  if (mPhiEnabled)
    pHistPhi->Reset();
  if (mEnergyEnabled && mThetaEnabled)
    pHistEnergyTheta->Reset(); 
  if (mEnergyEnabled && mPhiEnabled)
    pHistEnergyPhi->Reset();
  if (mThetaEnabled && mPhiEnabled)
    pHistThetaPhi->Reset();
  
  if (mAsciiFileEnabled) 
  {
    mAsciiFile.clear();
    mAsciiFile.seekp(0, std::ios::beg);
    mAsciiFile << "# Energy [MeV] | Theta [deg] | Phi [deg] | Flunece [particles/cm^2]\n";
  }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateTLFluenceDistributionActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateTLFluenceDistributionActor -- Begin of Run\n");
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLFluenceDistributionActor::UserSteppingAction(const GateVVolume*, const G4Step * step)
{
  // Find the lengt of the step
  G4double stepLength = step->GetStepLength();
  
  // Find the average energy along the step. 
  G4double energy = step->GetPreStepPoint()->GetKineticEnergy() - step->GetTotalEnergyDeposit() / 2.0; 
  
  // Get the direction of the step in the local coordiate system
  G4ThreeVector d = GetLocalDirection(step);
  
  // Calculate the theta and phi angle
  G4double theta = d.getTheta()*180.0/CLHEP::pi;
  G4double phi = d.getPhi()*180.0/CLHEP::pi;
  
  // Get the weight of the step
  G4double weight = step->GetPreStepPoint()->GetWeight();
  
  // Calculate the weighted fluence. In units cm^-2
  G4double dF = weight*stepLength/detectorVolume *cm*cm;
  
  // Store the results in the histograms
  storeStepInHistograms(energy,theta,phi,dF);
}

void GateTLFluenceDistributionActor::storeStepInHistograms(G4double energy, G4double theta, G4double phi, G4double dF)
{
  // Create the enabled histograms
  if (mEnergyEnabled)
  {
	pHistEnergy->Fill(energy,dF);
	
	if (mThetaEnabled)
	  pHistEnergyTheta->Fill(energy,theta,dF); 

	if (mPhiEnabled)
	  pHistEnergyPhi->Fill(energy,phi,dF);	
  }
  if (mThetaEnabled)
  {
	pHistTheta->Fill(theta,dF);
	if (mPhiEnabled)
	  pHistThetaPhi->Fill(theta,phi,dF); 
  }
  if (mPhiEnabled)
	pHistPhi->Fill(phi,dF);
  
  if (mAsciiFileEnabled) 
    mAsciiFile << energy << " " << theta << " " << phi << " " << dF << std::endl;

}

G4ThreeVector GateTLFluenceDistributionActor::GetLocalDirection(const G4Step *step)
{
  // Is this correct for nested structures??
  G4StepPoint* preStepPoint = step->GetPreStepPoint();
  G4TouchableHandle theTouchable = preStepPoint->GetTouchableHandle();
  G4ThreeVector p1 = preStepPoint->GetPosition();
  G4ThreeVector lp1 = theTouchable->GetHistory()->GetTopTransform().TransformPoint(p1);
  G4ThreeVector p2 = step->GetPostStepPoint()->GetPosition();
  G4ThreeVector lp2 = theTouchable->GetHistory()->GetTopTransform().TransformPoint(p2);
  
  return lp2-lp1;
}

//-----------------------------------------------------------------------------







#endif
