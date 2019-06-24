/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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

  mIsTrackAverageDEDX=false;
  mIsTrackAverageEdepDX=false;
  mIsDoseAverageDEDX=false;
  mIsDoseAverageEdepDX=false;
  mIsAverageKinEnergy=false;
  mIsGqq0EBT31stOrder=false;
  mIsGqq0EBT34thOrder=false;

  mIsLETtoWaterEnabled = false;
  mIsParallelCalculationEnabled = false;
  mAveragingType = "DoseAverage";
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
  // material for dedx computation for LETtoWater.
  G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);


  if (mAveragingType == "DoseAveraged" || mAveragingType == "DoseAverage" || mAveragingType == "doseaverage" || mAveragingType == "dose"){mIsDoseAverageDEDX = true;}
  else if (mAveragingType == "DoseAveragedEdep" || mAveragingType == "DoseAverageEdep" ){mIsDoseAverageEdepDX = true;}
  else if (mAveragingType == "TrackAveraged" || mAveragingType == "TrackAverage" || mAveragingType == "Track" || mAveragingType == "track" || mAveragingType == "TrackAveragedDXAveraged"){mIsTrackAverageDEDX = true;}
  else if (mAveragingType == "TrackAveragedEdep" || mAveragingType == "TrackAverageEdep" ){mIsTrackAverageEdepDX = true;}
  else if (mAveragingType == "AverageKinEnergy"){mIsAverageKinEnergy = true;}
  else if (mAveragingType == "gqq0EBT3linear"){mIsGqq0EBT31stOrder = true;mIsLETtoWaterEnabled=true;mIsDoseAverageDEDX = true;}
  else if (mAveragingType == "gqq0EBT3fourth"){mIsGqq0EBT34thOrder = true;mIsLETtoWaterEnabled=true;mIsDoseAverageDEDX = true;}
  else {GateError("The LET averaging Type" << GetObjectName()
                  << " is not valid ...\n Please select 'DoseAveraged' or 'TrackAveraged')");}

  // Output Filename
  mLETFilename = mSaveFilename;
  if (mIsDoseAverageDEDX)
    {
      mLETFilename= removeExtension(mSaveFilename) + "-doseAveraged."+ getExtension(mSaveFilename);
    }
  else if (mIsTrackAverageDEDX)
    {
      mLETFilename= removeExtension(mSaveFilename) + "-trackAveraged."+ getExtension(mSaveFilename);
    }

  if (mIsGqq0EBT31stOrder){
    mLETFilename= removeExtension(mLETFilename) + "-gqqZerolinear."+ getExtension(mLETFilename);
  }
  else if (mIsGqq0EBT34thOrder){
    mLETFilename= removeExtension(mLETFilename) + "-gqqZerofourthOrder."+ getExtension(mLETFilename);
  }
  else if (mIsLETtoWaterEnabled){
    mLETFilename= removeExtension(mLETFilename) + "-letToWater."+ getExtension(mLETFilename);
  }
  if (mIsAverageKinEnergy){
    mLETFilename= removeExtension(mLETFilename) + "-kinEnergyFluenceAverage."+getExtension(mLETFilename);
  }
  if (mIsParallelCalculationEnabled)
    {
      numeratorFileName= removeExtension(mLETFilename) + "-numerator."+ getExtension(mLETFilename);
      denominatorFileName= removeExtension(mLETFilename) + "-denominator."+ getExtension(mLETFilename);
    }

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mWeightedLETImage);
  SetOriginTransformAndFlagToImage(mNormalizationLETImage);
  SetOriginTransformAndFlagToImage(mDoseTrackAverageLETImage);

  // Resize and allocate images
  mWeightedLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mWeightedLETImage.Allocate();
  mNormalizationLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mNormalizationLETImage.Allocate();
  mDoseTrackAverageLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mDoseTrackAverageLETImage.Allocate();

  // Step Hit Type
  mStepHitType = RandomStepHitType ;// PostStepHitType; // Warning

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

//if (mIsGqq0EBT31stOrder || mIsGqq0EBT34thOrder)
//{

  if ((mIsGqq0EBT31stOrder) || (mIsGqq0EBT34thOrder))
  {  
    double ebt3_a0 = 1.0258;
    double ebt3_a1 = -0.0211;

    double ebt3_b0 = 1.0054;
    double ebt3_b1 = -6.4262E-4;
    double ebt3_b2 = -4.9426E-3;
    double ebt3_b3 = 4.1747E-4;
    double ebt3_b4 = -1.1622E-5;
        // ==========================
        // Note: gqq0 = 1/RE; and RE = a0 + a1*LETd ; therefore numerator and denomiator change in gqq
        // ==========================
           
           GateImageDouble::const_iterator iter_LET = mWeightedLETImage.begin();
           GateImageDouble::const_iterator iter_Edep = mNormalizationLETImage.begin();
           GateImageDouble::iterator iter_Final = mDoseTrackAverageLETImage.begin();
           
            if (mIsParallelCalculationEnabled) 
                {
                   for(iter_LET = mWeightedLETImage.begin(); iter_LET != mWeightedLETImage.end(); iter_LET++) {
                       //if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
                       //else 
                        *iter_Final = ebt3_a0 * (*iter_Edep) + ebt3_a1 * (*iter_LET);
                       iter_Edep++;
                       iter_Final++;
                    }
                    mNormalizationLETImage.Write(numeratorFileName);
                    mDoseTrackAverageLETImage.Write(denominatorFileName);
                    
               }
                     
               else
               {
                    for(iter_LET = mWeightedLETImage.begin(); iter_LET != mWeightedLETImage.end(); iter_LET++) {
                        //G4cout << *iter_LET <<  G4endl;
                        //G4cout << *iter_Edep <<  G4endl;
                         if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
                         else 
                         { 
                             if (mIsGqq0EBT31stOrder) *iter_Final = 1 / (ebt3_a0 + ebt3_a1 * (*iter_LET)/(*iter_Edep));
                             else 
                             {
                                 double let_voxel = (*iter_LET)/(*iter_Edep);
                                 double RE = (ebt3_b0 + ebt3_b1 * let_voxel  + ebt3_b2 * std::pow(let_voxel,2) + ebt3_b3 * std::pow(let_voxel,3) + ebt3_b4 * std::pow(let_voxel,4) );
                                 *iter_Final = 1 / RE;
                                 //G4cout << "4th: "<< *iter_Final << " -- 1st: "<<   (ebt3_a0 + ebt3_a1 * (*iter_LET)/(*iter_Edep)) << G4endl;
                                 
                             }
                            }
                        //G4cout <<"gqq: "<< *iter_Final <<  G4endl;
                       
                       iter_Edep++;
                       iter_Final++;
                    }
                    mDoseTrackAverageLETImage.Write(mLETFilename);
                }
           
     
    }
   else
   {


      if (mIsParallelCalculationEnabled) {
          
            mWeightedLETImage.Write(numeratorFileName);
            mNormalizationLETImage.Write(denominatorFileName);
        
      }
      else
        {
          GateImageDouble::const_iterator iter_LET = mWeightedLETImage.begin();
          GateImageDouble::const_iterator iter_Edep = mNormalizationLETImage.begin();
          GateImageDouble::iterator iter_Final = mDoseTrackAverageLETImage.begin();
          for(iter_LET = mWeightedLETImage.begin(); iter_LET != mWeightedLETImage.end(); iter_LET++) {
            if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
            else 
            {
                
                
                //if (mIsGqq0EBT31stOrder)
                //{
                    //*iter_Final = 1 / (ebt3_a0 + ebt3_a1 * (*iter_LET)/(*iter_Edep));
                    //G4cout << *iter_Final <<  G4endl;
                //}
                //else if (mIsGqq0EBT34thOrder)
                //{
                    //*iter_Final = 1/(ebt3_b0 + ebt3_b1 * (*iter_LET)/(*iter_Edep) + ebt3_b2 * std::pow((*iter_LET)/(*iter_Edep),2) + ebt3_b3 * std::pow((*iter_LET)/(*iter_Edep),3) + ebt3_b4 * std::pow((*iter_LET)/(*iter_Edep),4) );
                //}
                //else
                //{
                    *iter_Final = (*iter_LET)/(*iter_Edep);
                //}
                
            }
            iter_Edep++;
            iter_Final++;
          }
          mDoseTrackAverageLETImage.Write(mLETFilename);

    }
   }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::ResetData() {
  mWeightedLETImage.Fill(0.0);
  mNormalizationLETImage.Fill(0.0);

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
  //	G4cout << "In LET actor: " << step->GetTrack()->GetDefinition()->GetAtomicNumber() << G4endl;

  // Get edep and current particle weight
  const double weight = step->GetTrack()->GetWeight();

  // A.Resch tested calculation method:
  G4double edep = step->GetTotalEnergyDeposit();

  G4double steplength = step->GetStepLength();

  //if no energy is deposited or energy is deposited outside image => do nothing
  if (edep == 0) {
    GateDebugMessage("Actor", 5, "GateLETActor edep == 0 : do nothing\n");
    return;
  }
  if (index < 0) {
    GateDebugMessage("Actor", 5, "GateLETActor pixel index < 0 : do nothing\n");
    return;
  }

  const G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName();
  G4double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
  G4double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
  G4double energy=(energy1+energy2)/2;
  const G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();

  // Compute the dedx for the current particle in the current material
  double weightedLET =0;
  G4double dedx = emcalc->ComputeElectronicDEDX(energy, partname, material);
  // SPR to water is unity, but is overwritten if LET to water is enabled
  G4double SPR_ToWater =1.0;
  
  if (mIsLETtoWaterEnabled){
    G4double dedx_Water = emcalc->ComputeTotalDEDX(energy, partname->GetParticleName(), "G4_WATER") ;
    
    if ((dedx > 0) && (dedx_Water >0 ))
    {
        SPR_ToWater = dedx_Water/dedx;
        edep *=SPR_ToWater;
        dedx *=SPR_ToWater;
    }
    
  }

  double normalizationVal = 0;
  if (mIsDoseAverageDEDX) {
    weightedLET=edep*dedx*weight; // /(density/(g/cm3));
    normalizationVal = edep*weight;
  }
  else if (mIsTrackAverageDEDX) {
    weightedLET=dedx*steplength*weight;
    normalizationVal = steplength*weight;
  }
  else if (mIsTrackAverageEdepDX) {
    weightedLET=edep*weight;
    normalizationVal = steplength*weight;
  }
  else if (mIsDoseAverageEdepDX) {
    weightedLET=edep*edep/steplength*weight;
    normalizationVal = edep*weight;
  }
  else if (mIsAverageKinEnergy) {
    weightedLET=steplength*energy*weight;
    normalizationVal = steplength*weight;
  }

  //if (mIsLETtoWaterEnabled){
    //weightedLET = (weightedLET/dedx)*	emcalc->ComputeTotalDEDX(energy, partname->GetParticleName(), "G4_WATER") ;
  //}

  mWeightedLETImage.AddValue(index, weightedLET);
  mNormalizationLETImage.AddValue(index, normalizationVal);

  GateDebugMessageDec("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
