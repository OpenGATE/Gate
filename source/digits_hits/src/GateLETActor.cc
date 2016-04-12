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
  
  mCurrentEvent=-1;
  mIsRestrictedFlag = false;
   mIsTrackAveragedFluenceAveraged=false;
  mIsTrackAveragedDXAveraged=false;
  mIsTrackAveragedDXAveragedCancelled=false;
  mIsDoseAveragedDEDXAveraged=false;
  mIsTrackAveragedFluenceTrackAveraged = false;
  mIsDoseAveragedEdepDXAveraged=false;
  
  mIsRelUncertEnabled = false;
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
  //G4cout<<"=====================================================------------------======="<<G4endl;
  //G4cout<< "Averaging type is:" <<G4endl;
  //G4cout<<mAveragingType<<G4endl<<G4endl;

  if (mAveragingType == "DoseAveraged"){mIsDoseAveragedDEDXAveraged = true;}
  else if (mAveragingType == "TrackAveragedFluenceStep"){mIsTrackAveragedFluenceAveraged = true;}
  else if (mAveragingType == "TrackAveragedCancelled"){mIsTrackAveragedDXAveragedCancelled = true;}
  else if (mAveragingType == "TrackAveragedFluenceTrack"){mIsTrackAveragedFluenceTrackAveraged = true;}
  else if (mAveragingType == "TrackAveraged" || mAveragingType == "TrackAveragedDXAveraged"){mIsTrackAveragedDXAveraged = true;}
  else if (mAveragingType == "DoseAveragedEdep"){mIsDoseAveragedEdepDXAveraged = true;}
  
  else {GateError("The LET averaging Type" << GetObjectName()
              << " is not valid ...\n Please select 'DoseAveraged' or 'TrackAveraged')");}
//if (mIsTrackAveragedDXAveraged) {G4cout<<"yes yes track averaged DX"<<G4endl;}
//if (mIsTrackAveragedDXAveraged) {G4cout<<"yes yes track averaged DX"<<G4endl;}
  // Output Filename
  mLETFilename = mSaveFilename;

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mLETImage);
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mFinalImage);
  SetOriginTransformAndFlagToImage(mLETSecondMomentImage);
  SetOriginTransformAndFlagToImage(mLETUncertaintyFinalImage);
  SetOriginTransformAndFlagToImage(mRelUncertImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);
  SetOriginTransformAndFlagToImage(mLETTempImage);
  SetOriginTransformAndFlagToImage(mNumberOfHitsImage);

  // Resize and allocate images
  mLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLETImage.Allocate();
  mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mEdepImage.Allocate();
  mFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mFinalImage.Allocate();
  if (mIsRelUncertEnabled){
	  mRelUncertImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mRelUncertImage.Allocate();
  }
  if (mIsTrackAveragedFluenceTrackAveraged) {
	  mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mLastHitEventImage.Allocate();
	  
	  mLETTempImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mLETTempImage.Allocate();
	  
	  mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mNumberOfHitsImage.Allocate();
	}
  
  
    ////  mEdepImage.SetLastHitEventImage(&mLastHitEventImage);
    //mTestImage.EnableSquaredImage(false);
    //mTestImage.EnableUncertaintyImage(true);
    //// Force the computation of squared image if uncertainty is enabled
    ////if (mIsEdepUncertaintyImageEnabled) mEdepImage.EnableSquaredImage(true);
    //mTestImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    //mTestImage.Allocate();
    //mTestImage.SetFilename(removeExtension(mLETFilename) + "-testImage."+ getExtension(mLETFilename));
  
  if (mIsLETUncertaintyImageEnabled){
	  mLETSecondMomentImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mLETSecondMomentImage.Allocate();
	  mLETUncertaintyFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	  mLETUncertaintyFinalImage.Allocate();
  }
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
  if (mIsRelUncertEnabled){
	    mRelUncertImage.Write(removeExtension(mLETFilename) + "-relUncert."+ getExtension(mLETFilename));
	}
	
   //mTestImage.SaveData(mCurrentEvent+1, false);
  // Final computation: divide the cumulated LET by the cumulated
  // edep.
  if (mIsParallelCalculationEnabled){
	  mLETImage.Write(removeExtension(mLETFilename) + "-numerator."+ getExtension(mLETFilename));
	  mEdepImage.Write(removeExtension(mLETFilename) + "-denominator."+ getExtension(mLETFilename));
	  if (mIsLETUncertaintyImageEnabled)
		{
			mLETSecondMomentImage.Write(removeExtension(mLETFilename) + "-variance-unnormalizedSecondMoment."+ getExtension(mLETFilename));
		}
			
  }
  else
  {
	  
	  GateImageDouble::const_iterator iter_LET = mLETImage.begin();
	  GateImageDouble::const_iterator iter_Edep = mEdepImage.begin();
	  GateImageDouble::iterator iter_Final = mFinalImage.begin();
	  for(iter_LET = mLETImage.begin(); iter_LET != mLETImage.end(); iter_LET++) {
		if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
		else *iter_Final = (*iter_LET)/(*iter_Edep);
		iter_Edep++;
		iter_Final++;
	  }
	  mFinalImage.Write(mLETFilename);
 
  uncertaintyFilename = removeExtension(mLETFilename) + "-variance."+ getExtension(mLETFilename);
  if (mIsLETUncertaintyImageEnabled)
  {
	  GateImageDouble::const_iterator iter_LET_u = mLETImage.begin();
	  GateImageDouble::const_iterator iter_LET_secMoment = mLETSecondMomentImage.begin();
	  GateImageDouble::const_iterator iter_Edep_u = mEdepImage.begin();
	  GateImageDouble::iterator iter_Final_uncert = mLETUncertaintyFinalImage.begin();
	  //int abc = 0;
	  for(iter_LET_u = mLETImage.begin(); iter_LET_u != mLETImage.end(); iter_LET_u++) {
		if (*iter_Edep_u == 0.0) *iter_Final_uncert = 0.0; // do not divide by zero
		else *iter_Final_uncert = (*iter_LET_secMoment)/(*iter_Edep_u) - (*iter_LET_u)*(*iter_LET_u)/(*iter_Edep_u)/(*iter_Edep_u);
		iter_Edep_u++;
		iter_LET_secMoment++;
		iter_Final_uncert++;
		//abc++;
		//if (abc==20) {G4cout<<*iter_Final_uncert<<G4endl;}
	  }
	  mLETUncertaintyFinalImage.Write(uncertaintyFilename);
  
  }
  }
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::ResetData() {
  mLETImage.Fill(0.0);
  mEdepImage.Fill(0.0);
  
  
  if (mIsLETUncertaintyImageEnabled){
	  mLETSecondMomentImage.Fill(0.0);
  }
  
  if (mIsRelUncertEnabled) {
	  mRelUncertImage.Fill(0.0);
  }
  
  if (mIsTrackAveragedFluenceTrackAveraged) {
		  
	  mLETTempImage.Fill(0.0);
	  mNumberOfHitsImage.Fill(0);
	  mLastHitEventImage.Fill(0);
  }
  
  //mTestImage.Reset();
  //mFinalImage.Reset();
  
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
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateLETActor -- Begin of Event: " << mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel - begin\n");
  GateDebugMessageInc("Actor", 4, "enedepo = " << step->GetTotalEnergyDeposit() << Gateendl);
  GateDebugMessageInc("Actor", 4, "weight = " <<  step->GetTrack()->GetWeight() << Gateendl);
 
 
  
  // Get edep and current particle weight
  const double weight = step->GetTrack()->GetWeight();
  
  // A.Resch tested calculation method:
  const double edep = step->GetTotalEnergyDeposit()*weight;

  double steplength = step->GetStepLength();


   //if no energy is deposited or energy is deposited outside image => do nothing
  if (edep == 0) {
    GateDebugMessage("Actor", 5, "GateLETActor edep == 0 : do nothing\n");
    return;
  }
  if (index < 0) {
    GateDebugMessage("Actor", 5, "GateLETActor pixel index < 0 : do nothing\n");
    return;
  }
  // Get somes values
  double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
  //const G4VProcess * thisProcess= step->GetPostStepPoint()->GetProcessDefinedStep(); // ->GetProcessName() or ->GetProcessType() 
   //G4ProcessType thisProcessType = thisProcess->GetProcessType();
   //G4String thisProcessName;
   //thisProcessName = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
					
   //G4cout<<" Process name: "<< thisProcessName<<G4endl;
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName();
  double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
  double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
  double energy=(energy1+energy2)/2;
  const G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();
  
  
  
  
  //const double scoringVal=emcalc->ComputeElectronicDEDX(energy, partname, material);
  //mTestImage.AddValue(index, scoringVal);
  //mTestImage.AddValueAndUpdate(index, scoringVal);
  
  
  
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
  if (mIsDoseAveragedDEDXAveraged){
	  doseAveragedLET=edep*dedx; // /(density/(g/cm3)); 
	  normalizationVal = edep;
	  
	  //G4cout<<"This is dose Averaged: "<< doseAveragedLET<<G4endl;	
      //G4cout<< "LET t should not enter here mIsDoseAveraged" << G4endl;
  }

  else if (mIsTrackAveragedFluenceTrackAveraged){
	  //if (index > 8 && index < 10) {
	  //G4cout<<"mCurrentEvent: "<<mCurrentEvent<<G4endl;
	  //G4cout<<"mIndex: "<<index<<G4endl;
	    if ( mLastHitEventImage.GetValue(index) == mCurrentEvent){
			mLETTempImage.AddValue(index, dedx);
			mNumberOfHitsImage.AddValue(index, 1);
			//G4cout<<"lastHit = mCurrentEvent: "<<G4endl;
		}
		else {
			
			if (mNumberOfHitsImage.GetValue(index) > 0)
			{
				doseAveragedLET = (double) mLETTempImage.GetValue(index)/mNumberOfHitsImage.GetValue(index);
				//G4cout<<"+++++++++++++++++ saving value "<<G4endl;
				//G4cout<<"numberOfHitsImage >0: "<< mNumberOfHitsImage.GetValue(index)<<G4endl;
			}
			else { doseAveragedLET = 0.0;}
			//else {
				//doseAveragedLET = dedx;
				//G4cout<<"numberOfHits: "<< mNumberOfHitsImage.GetValue(index)<<G4endl;
			//}
			
			//G4cout<<"This is dose Averaged: "<< doseAveragedLET<<G4endl;
			
			
			
			if (doseAveragedLET > 0.0001){
				//G4cout<<"Dose Averaging: "<< doseAveragedLET<<G4endl;
				//G4cout<<"here is actually correctly saved "<<G4endl;
				mLETImage.AddValue(index, doseAveragedLET);
				// Store the Edep (needed for final computation)
				mEdepImage.AddValue(index, 1.0);
			}
			
			mLETTempImage.SetValue(index,dedx);
			mNumberOfHitsImage.SetValue(index,1);
	
		}
		//G4cout<<"mLETImage: " << mLETImage.GetValue(index)<< G4endl;
		//G4cout <<"mdenominator: " << mEdepImage.GetValue(index)<< G4endl;
		//G4cout <<"globalLET: " <<mLETImage.GetValue(index)/mEdepImage.GetValue(index)<< G4endl<< G4endl;
	//}
	    mLastHitEventImage.SetValue(index, mCurrentEvent);
  }
  else if (mIsTrackAveragedFluenceAveraged){
	  // this implementation probably varies with production threshold:
	  //doseAveragedLET=edep;
	  // hopefully no production threshold problem
	  //doseAveragedLET=dedx*steplength*weight;
	  //normalizationVal = steplength;
	  
	  //doseAveragedLET+=steplength;
	  //normalizationVal+= 1;
	  //G4cout<<"This is Track Averaged Track!!: "<< doseAveragedLET<<G4endl;
	  doseAveragedLET = weight*(dedx);
	 
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
  
  
  else if (mIsTrackAveragedDXAveraged)
  { 
	   //G4cout<< "LET t should  enter here TrackAveragedDXAveraged" << G4endl;
	  doseAveragedLET=dedx*steplength;
	  normalizationVal = steplength;
	  //doseAveragedLET=edep;
	  //normalizationVal = steplength;
  }  
  else if (mIsTrackAveragedDXAveragedCancelled)
  { 
	   //G4cout<< "LET t should not enter here TrackAveragedCancelled" << G4endl;
	   
	  doseAveragedLET=edep;
	  normalizationVal = steplength;
	  //G4cout<<"Edep : " <<edep<<G4endl;
	  //G4cout<<"DX  : "<< steplength<<G4endl;
	  //G4cout<<"Edep : " <<doseAveragedLET<<G4endl;
	  //G4cout<<"DX  : "<< normalizationVal<<G4endl;
  }
  ////"DoseAveragedEdep"){mIsTrackAveraged = false; mIsDoseAveraged = false; }
  else if (mIsDoseAveragedEdepDXAveraged)
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
			 
  }
  
   
  
  
  

  if (mIsLETUncertaintyImageEnabled)
  {
	   //G4cout<< "LET t should not enter here mIsLETUncertaintyImageEnabled" << G4endl;
	double secondMomentLET;
	if (mIsDoseAveragedDEDXAveraged){
		secondMomentLET = edep*dedx*dedx;
	}
	else if (mIsTrackAveragedDXAveraged) { secondMomentLET = steplength*dedx*dedx;}
	else if (mIsTrackAveragedFluenceAveraged) { secondMomentLET = dedx*dedx;}
	else if (mIsTrackAveragedDXAveragedCancelled) { secondMomentLET = edep*edep/steplength;}
	else if (mIsDoseAveragedEdepDXAveraged) { secondMomentLET = edep*edep/steplength;}
    mLETSecondMomentImage.AddValue(index, secondMomentLET);
  } 
  //G4cout<< "difference between LETT and edep"<< doseAveragedLET-edep << G4endl;
  /*
  G4cout<<"This is dedx from emcalc: "<<dedx<<G4endl;
  G4cout<<"This is edep/dx: "<<edep/steplength<<G4endl;
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
  if (mIsRelUncertEnabled){

	  if (doseAveragedLET > 0 )
	  {
		  //G4cout<<"Old value: " <<mLETImage.GetValue(index)/mEdepImage.GetValue(index)<<G4endl;
		  //G4cout<<"New value: " <<doseAveragedLET/normalizationVal<<G4endl;
		  //G4cout<<"diff: "  <<(mLETImage.GetValue(index)/mEdepImage.GetValue(index)-doseAveragedLET/normalizationVal)<<G4endl;
		
		  double L_i = mLETImage.GetValue(index);
		  double n_i = mEdepImage.GetValue(index);
		  if (n_i>0){
		double relUncert = 1-(L_i/(L_i+doseAveragedLET)) *((n_i+normalizationVal)/n_i) ;
		 double oldRelUncertVal = mRelUncertImage.GetValue(index);
			   if (mCurrentEvent % 10000 == 0)
	   {
		   	mRelUncertImage.SetValue(index, relUncert);
	   }
	   else if (abs(oldRelUncertVal) < relUncert)
	   { mRelUncertImage.SetValue(index, relUncert);}
	
		  //G4cout<<"Li : "  << (L_i)<<G4endl;
		  //G4cout<<"ni : "  << (n_i)<<G4endl;
		  //G4cout<<"deltaL : "  << (doseAveragedLET)<<G4endl;
		  //G4cout<<"deltaN : "  << (normalizationVal)<<G4endl;
		  //G4cout<<"Li / li + delta: "  << (L_i/(L_i+doseAveragedLET))<<G4endl;
		  //G4cout<<" ni + delta/ni: "  << ((n_i+normalizationVal)/n_i)<<G4endl;
		  //G4cout<<" L * n: "  << (L_i/(L_i+doseAveragedLET)) *((n_i+normalizationVal)/n_i)<<G4endl;
		  //G4cout<<" relUncert: "  << relUncert<<G4endl;
				  //G4cout<<"relChange: " <<mRelUncertImage.GetValue(index)<<G4endl<<G4endl;
			  }
	  }
	  //else {mRelUncertImage.SetValue(index,0);}
  }
  // Store the LET
    
  if (!mIsTrackAveragedFluenceTrackAveraged)
	{
		//G4cout<<"should not enter here:  dose Averaged: "<< doseAveragedLET<<G4endl;
		  mLETImage.AddValue(index, doseAveragedLET);

		// Store the Edep (needed for final computation)
		mEdepImage.AddValue(index, normalizationVal);
	}
   //double zV= step->GetPreStepPoint()->GetPosition().z(); //zV>34 && zV<35.0
   /*
  if ((mCurrentEvent % 1 == 0) && index==20){
	    double iter_LET_u =mLETImage.GetValue(index);
   double iter_Edep_u =mEdepImage.GetValue(index);
   double iter_LET_secMoment =  mLETSecondMomentImage.GetValue(index);
   
  G4cout<<"====================================================="<<G4endl;
   G4cout<<"av LET numerator: "<< iter_LET_u<<G4endl;
  G4cout<<"av LET denominator: "<< iter_Edep_u<<G4endl;
  G4cout<<"av LET var numerator: "<< iter_LET_secMoment<<G4endl;

  double finalVar = (iter_LET_secMoment)/(iter_Edep_u) - (iter_LET_u)*(iter_LET_u)/(iter_Edep_u)/(iter_Edep_u);
  double secMoment = (iter_LET_secMoment)/(iter_Edep_u) ;
  double firstMomentSquared = (iter_LET_u)*(iter_LET_u)/(iter_Edep_u)/(iter_Edep_u);
	 G4cout<<"first Moment: "<< (iter_LET_u)/(iter_Edep_u)<<G4endl;
  G4cout<<"sec Moment: "<< secMoment<<G4endl;
  G4cout<<"first Moment squared: "<< firstMomentSquared<<G4endl;
  G4cout<<"final variance: "<< finalVar<<G4endl;
  
 G4cout<<"This is dedx from emcalc: "<<dedx<<G4endl;
  G4cout<<"This is steplength: "<<steplength/(um)<<" um"<<G4endl;
  
  G4cout<<"This is steplength*dedx: "<<steplength*dedx<<G4endl;
  G4cout<<"This is steplength*dedx*dedx: "<<steplength*dedx*dedx<<G4endl<<G4endl;
  ////G4cout<<"This is x: "<<step->GetPreStepPoint()->GetPosition().x()<< " keV"<<G4endl;
  ////G4cout<<"This is y: "<<step->GetPreStepPoint()->GetPosition().y()<< " keV"<<G4endl;
  //G4cout<<"This is z: "<<step->GetPreStepPoint()->GetPosition().z()<< " keV"<<G4endl<<G4endl;
 
  
  } 
  */
  GateDebugMessageDec("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
