/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateLETActor :
  \brief
*/

// gate
#include "GateLETActor.hh"
#include "GateMiscFunctions.hh"

// g4
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateLETActor::GateLETActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateLETActor() -- begin\n");
  mIsRestrictedFlag = false;
  //mDeltaRestricted = 800000.0;
  mIsLETUncertaintyImageEnabled = false;
  mIsDoseToWaterEnabled = false;
  mIsParallelCalculationEnabled = false;
  mAveragingType = "DoseAveraged";
  pMessenger = new GateLETActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateLETActor() -- end\n");
  emcalc = new G4EmCalculator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateLETActor::~GateLETActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateLETActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateLETActor -- Construct - begin\n");
  GateVImageActor::Construct();
  
  
  // Find G4_WATER. This it needed here because we will used this
  // material for dedx computation for DoseToWater.
  G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  
  if (mAveragingType == "DoseAveraged"){mIsDoseAveraged = true; mIsTrackAveraged= false;}
  else if (mAveragingType == "TrackAveraged"){mIsTrackAveraged = true; mIsDoseAveraged = false; }
  else if (mAveragingType == "TrackAveragedCancelled"){mIsTrackAveraged = true; mIsDoseAveraged = true; }
  else if (mAveragingType == "DoseAveragedEdep"){mIsTrackAveraged = false; mIsDoseAveraged = false; }
  else {GateError("The LET averaging Type" << GetObjectName()
              << " is not valid ...\n Please select 'DoseAveraged' or 'TrackAveraged')");}

  // Output Filename
  mLETFilename = mSaveFilename;

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mLETImage);
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mFinalImage);
  SetOriginTransformAndFlagToImage(mLETSecondMomentImage);
  SetOriginTransformAndFlagToImage(mLETUncertaintyFinalImage);

  // Resize and allocate images
  mLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLETImage.Allocate();
  mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mEdepImage.Allocate();
  mFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mFinalImage.Allocate();
  
  
  mLETSecondMomentImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLETSecondMomentImage.Allocate();
  mLETUncertaintyFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLETUncertaintyFinalImage.Allocate();
  // Warning: for the moment we force to PostStepHitType. This is ok
  // (slightly faster) if voxel sizes are the same between the
  // let-actor and the attached voxelized volume. But wrong if not.
  mStepHitType = PostStepHitType;// RandomStepHitType; // Warning

  // Print information
  GateMessage("Actor", 1,
              "\tLET Actor      = '" << GetObjectName() << Gateendl <<
              "\tLET image      = " << mLETFilename << Gateendl <<
              "\tResolution     = " << mResolution << Gateendl <<
              "\tHalfSize       = " << mHalfSize << Gateendl <<
              "\tPosition       = " << mPosition << Gateendl);

  ResetData();
  GateMessageDec("Actor", 4, "GateLETActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save datamDeltaRestricted
void GateLETActor::SaveData() {
  GateVActor::SaveData();

  // Final computation: divide the cumulated LET by the cumulated
  // edep.
  
  if (mIsParallelCalculationEnabled){
	  mLETImage.Write(removeExtension(mLETFilename) + "-numerator."+ getExtension(mLETFilename));
	  mEdepImage.Write(removeExtension(mLETFilename) + "-denominator."+ getExtension(mLETFilename));
  }
  else
  {
	  
	  GateImage::const_iterator iter_LET = mLETImage.begin();
	  GateImage::const_iterator iter_Edep = mEdepImage.begin();
	  GateImage::iterator iter_Final = mFinalImage.begin();
	  for(iter_LET = mLETImage.begin(); iter_LET != mLETImage.end(); iter_LET++) {
		if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
		else *iter_Final = (*iter_LET)/(*iter_Edep);
		iter_Edep++;
		iter_Final++;
	  }
	  mFinalImage.Write(mLETFilename);
 }
  uncertaintyFilename = removeExtension(mLETFilename) + "-variance."+ getExtension(mLETFilename);
  if (mIsLETUncertaintyImageEnabled)
  {
	  GateImage::const_iterator iter_LET_u = mLETImage.begin();
	  GateImage::const_iterator iter_LET_secMoment = mLETSecondMomentImage.begin();
	  GateImage::const_iterator iter_Edep_u = mEdepImage.begin();
	  GateImage::iterator iter_Final_uncert = mLETUncertaintyFinalImage.begin();
	  for(iter_LET_u = mLETImage.begin(); iter_LET_u != mLETImage.end(); iter_LET_u++) {
		if (*iter_Edep_u == 0.0) *iter_Final_uncert = 0.0; // do not divide by zero
		else *iter_Final_uncert = (*iter_LET_secMoment)/(*iter_Edep_u) - (*iter_LET_u)*(*iter_LET_u)/(*iter_Edep_u)/(*iter_Edep_u);
		iter_Edep_u++;
		iter_LET_secMoment++;
		iter_Final_uncert++;
	  }
	  mLETUncertaintyFinalImage.Write(uncertaintyFilename);
  
  }
  
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::ResetData() {
  mLETImage.Fill(0.0);
  mEdepImage.Fill(0.0);
  mLETSecondMomentImage.Fill(0.0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateLETActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at each event
void GateLETActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  GateDebugMessage("Actor", 3, "GateLETActor -- Begin of Event: " << mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel - begin\n");
  GateDebugMessageInc("Actor", 4, "enedepo = " << step->GetTotalEnergyDeposit() << Gateendl);
  GateDebugMessageInc("Actor", 4, "weight = " <<  step->GetTrack()->GetWeight() << Gateendl);

  // Get edep and current particle weight
  double weight = step->GetTrack()->GetWeight();
  
  // A.Resch tested calculation method:
  double edep = step->GetTotalEnergyDeposit()*weight;

  double steplength = step->GetStepLength();


  // if no energy is deposited or energy is deposited outside image => do nothing
  //if (edep == 0) {
    //GateDebugMessage("Actor", 5, "GateLETActor edep == 0 : do nothing\n");
    //return;
  //}
  //if (index < 0) {
    //GateDebugMessage("Actor", 5, "GateLETActor pixel index < 0 : do nothing\n");
    //return;
  //}

  // Get somes values
  double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
  G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName();
  double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
  double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
  double energy=(energy1+energy2)/2;
  G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();
  //double voxelVolume=mLETImage.GetVoxelVolume();
  // The following variable should be used: mIsRestrictedFlag mDeltaRestricted

  // Compute the dedx for the current particle in the current material
  double doseAveragedLET =0;
  //G4int doseAveragedLETG4int =0;
  G4double dedx = emcalc->ComputeElectronicDEDX(energy, partname, material);
  //G4cout<<"dedx G4double: " << dedx <<G4endl;
  //G4int dedxInt = dedx/(keV/mm);
  //G4cout<<"dedx G4int: " << dedxInt <<G4endl;
    // A. Resch new calc method: 09.Feb.2016
  //G4double edep = dedx*steplength*weight;
/*  if (dedx>10){
  G4cout<<"this dx um "<<steplength/um<<G4endl;
  G4cout<<"this dx "<<steplength<<G4endl;
  G4cout<<"this edep"<< edep/keV<<G4endl;
  G4cout<<"this (edep/dx)/dx "<< (edep/keV)/(steplength/um)/(steplength/um)<<G4endl;
  G4cout<<"this DEDX/dx  um "<< (dedx/(keV/um))/(steplength/um) << G4endl;
  G4cout<<"this DEDX/dx   "<< dedx/(steplength) << G4endl;
  G4cout<<"this E*DEDX/E "<< edep*dedx/edep <<G4endl;
  G4cout<<"this DEDX "<< dedx <<G4endl;
  G4cout<<"this is unrestricted dedx:   " << emcalc->ComputeElectronicDEDX(energy, partname, material)<<G4endl;
  
  G4cout<<"new event"<<G4endl<<G4endl;
  
  }
  */
  //G4double meanFreePath = emcalc->GetMeanFreePath(energyOfSecElectron)
  //G4double 	GetMeanFreePath (G4double kinEnergy, const G4ParticleDefinition *, const G4String &processName, const G4Material *, const G4Region *r=0)
  if (mIsRestrictedFlag){
	   //G4cout<< "LET t should not enter here mIsRestrictedFlag" << G4endl;
	  // PDGcode 11 is an electron
	  // LET restricted is defined in that way that all secondary electrons above a threshold are removed (they carry energy away)
	  // see ICRU 85
	  if (partname->GetPDGEncoding() == 11 && step->GetTrack()->GetParentID() == 0)
	  {
		  dedx = emcalc->ComputeElectronicDEDX(energy, partname, material,mDeltaRestricted);
	  }
	  //G4cout<<"this is the cut value: " << mDeltaRestricted<<G4endl;
	  //G4cout<<"this is the cut value mm: " << mDeltaRestricted *mm<<G4endl;
	  //G4cout<<"this is the cut value um: " << mDeltaRestricted *um<<G4endl;
	  //G4cout<<"this is restricted dedx: " <<emcalc->ComputeElectronicDEDX(energy, partname, material,mDeltaRestricted)<<G4endl;
	  //G4cout<<"this is unrestricted dedx:   " << emcalc->ComputeElectronicDEDX(energy, partname, material)<<G4endl;
	  //G4cout<<"this is restricted dedx m: " <<emcalc->ComputeElectronicDEDX(energy, partname, material,mDeltaRestricted*m)<<G4endl;
	  //G4cout<<"this is restricted dedx mm: " <<emcalc->ComputeElectronicDEDX(energy, partname, material,mDeltaRestricted*mm)<<G4endl;
  }
  
  //G4cout<<"new event"<<G4endl<<G4endl;

  double normalizationVal = 0;
  //G4cout<<"This is edep: "<< edep<<G4endl;
  if (mIsDoseAveraged && !mIsTrackAveraged){
	  doseAveragedLET=edep*dedx; // /(density/(g/cm3)); 
	  normalizationVal = edep;
	  
	  //G4cout<<"This is dose Averaged: "<< doseAveragedLET<<G4endl;	
      //G4cout<< "LET t should not enter here mIsDoseAveraged" << G4endl;
  }
  else if (mIsTrackAveraged && !mIsDoseAveraged){
	  // this implementation probably varies with production threshold:
	  //doseAveragedLET=edep;
	  // hopefully no production threshold problem
	  //doseAveragedLET=dedx*steplength*weight;
	  //normalizationVal = steplength;
	  
	  //doseAveragedLET+=steplength;
	  //normalizationVal+= 1;
	  //G4cout<<"This is Track Averaged Track!!: "<< doseAveragedLET<<G4endl;
	  doseAveragedLET = weight*(dedx/(MeV/um));
	 
	  //if (doseAveragedLET > 0.000000001){
	  normalizationVal = weight;
	  
	  //doseAveragedLETG4int= weight*dedxInt;
	  //G4cout<<"doseAvLet numerator double: " << doseAveragedLET<<G4endl;
	  //G4cout<<"doseAvLet numerator int: " << doseAveragedLETG4int<<G4endl<<G4endl; 
	  
	   //}
	   ////else { normalizationVal = 0;}
	   //if (weight != 1 || partname->GetParticleName()!="proton" || material->GetName() != "Water") { 
		   
	  //G4cout <<"weig "<< weight <<G4endl;
	  //G4cout <<"dedx " << dedx << G4endl;
	  //G4cout <<"dx   " << steplength <<G4endl;
	  //G4cout << "partName " <<partname->GetParticleName()<<G4endl;
	  //G4cout << "matName " << material->GetName()<<G4endl;
	  //G4cout <<"Fraction " << doseAveragedLET/normalizationVal << G4endl;
	  //G4cout <<"L_d " << doseAveragedLET << G4endl;
	  //G4cout <<"Norm " << normalizationVal << G4endl<<G4endl;
		//}
  }
  
  
  // TrackAveragedCancelled   mIsTrackAveraged = true; mIsDoseAveraged = true; 
  else if (mIsTrackAveraged == true && mIsDoseAveraged == true)
  { 
	   //G4cout<< "LET t should not enter here TrackAveragedCancelled" << G4endl;
	  doseAveragedLET=dedx*steplength;
	  normalizationVal = steplength;
	  //doseAveragedLET=edep;
	  //normalizationVal = steplength;
  }
  ////"DoseAveragedEdep"){mIsTrackAveraged = false; mIsDoseAveraged = false; }
  else if (mIsTrackAveraged == false && mIsDoseAveraged == false)
  {
	   //G4cout<< "LET t should not enter here mIsTrackAveragedEdep" << G4endl;
	  doseAveragedLET=edep*edep/steplength;
	  normalizationVal = edep;
  }
  
  if (mIsDoseToWaterEnabled){
	   //G4cout<< "LET t should not enter here mIsDoseToWaterEnabled" << G4endl;
	  double energyScalingFactor= emcalc->ComputeTotalDEDX(energy, partname->GetParticleName(), "G4_WATER"); // include here also mDeltaRestricted
	  energyScalingFactor /= emcalc->ComputeTotalDEDX(energy, partname, material); // include here also mDeltaRestricted
	  double massScalingFactor = density;
	  massScalingFactor /= 1.0;
	  if (mIsDoseAveraged){
		  doseAveragedLET*=(massScalingFactor*energyScalingFactor*energyScalingFactor);
		  normalizationVal*=(massScalingFactor*energyScalingFactor);
	  }
	  else if (mIsTrackAveraged){
		  doseAveragedLET*=(energyScalingFactor);
	  }
			  
		//factor*=calc.ComputeTotalDEDX(aStep->GetPreStepPoint()->GetKineticEnergy(),proton,HU_10to4,1.0*km);

		//factor/=calc.ComputeTotalDEDX(aStep->GetPreStepPoint()->GetKineticEnergy(),proton,aStep->GetTrack()->GetStep()->GetPreStepPoint()->GetMaterial(),1.0*km);

		  //massScalingFactor= density;
		  //massScalingFactor/= HU_10to4->GetDensity();
  }
  if (mIsLETUncertaintyImageEnabled)
  {
	   //G4cout<< "LET t should not enter here mIsLETUncertaintyImageEnabled" << G4endl;
	double secondMomentLET;
	if (mIsDoseAveraged){
		secondMomentLET = edep*dedx*dedx;
	}
	else { secondMomentLET = steplength*dedx*dedx;}
    mLETSecondMomentImage.AddValue(index, secondMomentLET);
  } 
  //G4cout<< "difference between LETT and edep"<< doseAveragedLET-edep << G4endl;
  /*
  G4cout<<"This is dedx from emcalc: "<<dedx<<G4endl;
  G4cout<<"This is edep/dx: "<<dedx<<G4endl;
  G4cout<<"This is edep: "<<edep/(keV)<< " keV"<<G4endl;
  G4cout<<"This is steplength: "<<steplength/(um)<<" mum"<<G4endl;
  G4cout<<"This is density: "<<density*1.6e-19<<"  default*1.6e19"<<G4endl;
  G4cout<<"This is density: "<<density<<"  default"<<G4endl;
  G4cout<<"This is density: "<<density/(g/cm3)<<" g/cm3"<<G4endl<<G4endl;
  G4cout<<"This is voxelVolume: "<<voxelVolume/cm3<<" cm3"<<G4endl<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep*(dedx/(density*1.6e-19*voxelVolume))<<" 1.6e19"<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep*(dedx/(density*voxelVolume))<<" default"<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep*dedx/(density*voxelVolume)<<" timesVoxelVolume"<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep*dedx/(density/(g/cm3)*voxelVolume)<<" default, but only density scaled"<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep/(keV)*dedx/(keV/um)/(density/(g/cm3)*voxelVolume)<<" manually set"<<G4endl<<G4endl;
  double theDensity = density/(g/cm3);
  G4cout<<"This is theDensity: "<<theDensity<<" g/cm3"<<G4endl;
  G4cout<<"This is doseAveragedLET: "<< edep*dedx/(theDensity*voxelVolume)<<" theDensity"<<G4endl<<G4endl;
  */

  // Store the LET
  mLETImage.AddValue(index, doseAveragedLET);

  // Store the Edep (needed for final computation)
  mEdepImage.AddValue(index, normalizationVal);

  GateDebugMessageDec("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
