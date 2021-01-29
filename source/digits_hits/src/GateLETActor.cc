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
  mIsSwairApprox = false;
  mIsMeanEnergyToProduceIonPairInAir = false;
  mIsMeanEnergyToProduceIonPairInAirAR = false;
  mKGrosswendt = false;
  mIsLETtoWaterEnabled = false;
  mIsParallelCalculationEnabled = false;
  mAveragingType = "DoseAverage";
  mSetMaterial = "G4_WATER";
  k_FitParWAir = 0.08513;
  //mRestrictedLET = false;
  mCutVal = DBL_MAX ; // 10*keV;
  
  mLETthrMin = 0.;
  mLETthrMax = DBL_MAX; 
  
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
  G4cout << "Build material: " << mSetMaterial << G4endl;
  G4NistManager::Instance()->FindOrBuildMaterial(mSetMaterial);

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
  else if (mAveragingType == "massSprWaterAirApprox") {mIsSwairApprox = true;}
  else if (mAveragingType == "meanEnergyToProduceIonPairApproxDennis") { mIsMeanEnergyToProduceIonPairInAir = true; }
  else if (mAveragingType == "meanEnergyToProduceIonPairApproxGrosswendtAR") { mIsMeanEnergyToProduceIonPairInAirAR = true; }
  else if (mAveragingType == "meanEnergyToProduceIonPairApproxGrosswendt") { mIsMeanEnergyToProduceIonPairInAir = true; mKGrosswendt =true;}
  else {GateError("The LET averaging Type" << GetObjectName()
                  << " is not valid ...\n Please select 'DoseAveraged' or 'TrackAveraged')");}

  //if (mCutVal < DBL_MAX){  mRestrictedLET = true; }
        //const double k_dennis = 0.08513;
      //const double k_grosswendt = 0.05264;
   if (mKGrosswendt){
       k_FitParWAir = 0.05264; // this is the k value fitted to the grosswendt data; default value is set to the dennis data (0.08513)
   }
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
    mLETFilename= removeExtension(mLETFilename) + "-letTo" + mSetMaterial + "."+ getExtension(mLETFilename);
  }
  if (mIsAverageKinEnergy){
    mLETFilename= removeExtension(mLETFilename) + "-kinEnergyFluenceAverage."+getExtension(mLETFilename);
  }
  if (mIsSwairApprox ) {
      mLETFilename= removeExtension(mLETFilename) + "-massSPRwaterAirApprox."+getExtension(mLETFilename);
  }
  if (mIsMeanEnergyToProduceIonPairInAir ) {
      if (mKGrosswendt){
          
        mLETFilename= removeExtension(mLETFilename) + "-meanEProduceIonPairApproxGross."+getExtension(mLETFilename);
       }
       else {
           
        mLETFilename= removeExtension(mLETFilename) + "-meanEProduceIonPairApproxDennis."+getExtension(mLETFilename);
       }
  }  
   if (mIsMeanEnergyToProduceIonPairInAirAR ) {
        mLETFilename= removeExtension(mLETFilename) + "-meanEProduceIonPairApproxGrossAR."+getExtension(mLETFilename);
       }
  if (mCutVal < DBL_MAX){  
     mLETFilename= removeExtension(mLETFilename) + "-restricted."+getExtension(mLETFilename);
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
  mStepHitType = mStepHitType ; // RandomStepHitType ;// PostStepHitType; 
  
  
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
                         if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
                         else 
                         { 
                             if (mIsGqq0EBT31stOrder) *iter_Final = 1 / (ebt3_a0 + ebt3_a1 * (*iter_LET)/(*iter_Edep));
                             else 
                             {
                                 double let_voxel = (*iter_LET)/(*iter_Edep);
                                 double RE = (ebt3_b0 + ebt3_b1 * let_voxel  + ebt3_b2 * std::pow(let_voxel,2) + ebt3_b3 * std::pow(let_voxel,3) + ebt3_b4 * std::pow(let_voxel,4) );
                                 *iter_Final = 1 / RE;
                                
                             }
                            }
                       
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
                    *iter_Final = (*iter_LET)/(*iter_Edep);
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
  if (mStepHitType == PreStepHitType) {
       energy = energy1;
      }
  const G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();

  // Compute the dedx for the current particle in the current material
  double weightedLET =0;
  double normalizationVal = 0;
  
  G4double dedx = emcalc->ComputeElectronicDEDX(energy, partname, material,mCutVal);
  //if (mRestrictedLET){
      //dedx = emcalc->ComputeElectronicDEDX(energy, partname, material,mCutVal);
  //}
  // SPR to water is unity, but is overwritten if LET to water is enabled
  G4double SPR_ToWater =1.0;
  
  if (mIsLETtoWaterEnabled){
    G4double dedx_Water = emcalc->ComputeElectronicDEDX(energy, partname->GetParticleName(), mSetMaterial, mCutVal) ;
    
    //if (mRestrictedLET){
        //dedx_Water = emcalc->ComputeElectronicDEDX(energy, partname->GetParticleName(), mSetMaterial, mCutVal) ;
    //}
    
    if ((dedx > 0) && (dedx_Water >0 ))
    {
        SPR_ToWater = dedx_Water/dedx;
        edep *=SPR_ToWater;
        dedx *=SPR_ToWater;
    }
  }
  
    // max and min LET thresholds
  if ( dedx < mLETthrMin) {
      return;
  }
  if (dedx > mLETthrMax) {
      return;
  }


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
  else  if (mIsSwairApprox ) {
      const double a_con = 1.1425;
      const double b_con = 0.025;
      const double n_con = 0.0012;
      // avoid singularity if E approaches zero; assumes saturation
      if ( energy <= 1 ) {
          energy = 1;
      }
      //if ( energy <= b_con ) {
          //energy = b_con*1.1;
      //}
      weightedLET=steplength*weight * a_con*energy/(pow(energy-b_con , (1+n_con)));
      normalizationVal = steplength*weight;
      
  }
  else if (mIsMeanEnergyToProduceIonPairInAir) {
      const double weovere_con = 33.97;

      // avoid singularity if E approaches k;
      if ( energy <= 1 ) {
          energy = 1;
      }
      //if ( energy <= k_FitParWAir ) {
          //energy = k_FitParWAir*1.1;
      //}
      weightedLET=steplength*weight * weovere_con*energy/(energy-k_FitParWAir);
      normalizationVal = steplength*weight;
      
      
  }
  else if (mIsMeanEnergyToProduceIonPairInAirAR) {
      //double coeffs[] = {36.6957553908936,-0.898563335334212,0.0534009868181429,0.0790208004831028,-0.0274748605894538,0.00368798591275894,-0.000182298778521255};
      double coeffs[] = {36.7150819265373,-0.895836883500243,0.00751918283412532,0.0982132388275925,-0.0218536548447117,-0.000757742745394211,0.000654143801897059,-5.09569713537415e-05};
      int degPolyn = 7;
      double energyLog = std::log(energy);
      double poly = polynomial(coeffs,  degPolyn, energyLog);
       weightedLET=steplength*weight *poly;
      normalizationVal = steplength*weight;
      
  }

  mWeightedLETImage.AddValue(index, weightedLET);
  mNormalizationLETImage.AddValue(index, normalizationVal);

  GateDebugMessageDec("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------

double GateLETActor::polynomial(double * coefs, int deg, double x) {
    double factor = 1, result = 0; 
    for(int term = 0; term <= deg; term++) {
        result += coefs[term] * factor;
        factor *= x;
    }
    return result;
}
