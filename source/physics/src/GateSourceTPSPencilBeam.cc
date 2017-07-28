/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//=======================================================
//  Main contributors: Loïc Grevillot, David Boersma
//  General definition of the class
// This class allows for simulating treatment plans for Pencil Beam Scanning applications.
// The source need 2 inputs: a beam model of the system and a treatment plans
// It will simulate each single pencil beam of the treatment plan using the GateSourcePencilBeam class.
//=======================================================

#include "GateConfiguration.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <exception>
#include <sstream>
#include "GateSourceTPSPencilBeam.hh"
#include "G4Proton.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"

//------------------------------------------------------------------------------------------------------
//  try get N values of type T from a given input line
// * throw exception with informative error message in case of trouble.
// * NOTE that while this catches some common errors, it is not yet fool proof.
template<typename T, int N>
typename std::vector<T> parse_N_values_of_type_T(std::string line,int lineno, const std::string& fname){
  GateMessage("Beam", 5, "[TPSPencilBeam] trying to parse line " << lineno << " from file " << fname << Gateendl );
  std::istringstream iss(line);
  typename std::istream_iterator<T> iss_end;
  typename std::istream_iterator<T> isiT(iss);
  typename std::vector<T> vecT;
  while (isiT != iss_end) vecT.push_back(*(isiT++));
  int nread = vecT.size();
  if (nread != N){
    std::ostringstream errMsg;
    errMsg << "wrong number of values (" << nread << ") on line " << lineno << " of " << fname
           << ", expected " << N << " value(s) of type " << typeid(T).name() << std::endl;
    throw std::runtime_error(errMsg.str());
  }
  return vecT;
}
//------------------------------------------------------------------------------------------------------
// Function to read the next content line
// * skip all comment lines (lines string with a '#')
// * skip empty
// * throw exception with informative error message in case of missing data
std::string ReadNextContentLine( std::istream& input, int& lineno, const std::string& fname ) {
  while ( input ){
    std::string line;
    std::getline(input,line);
    ++lineno;
    if (line.empty()) continue;
    if (line[0]=='#') continue;
    return line;
  }
  throw std::runtime_error(std::string("reached end of file '")+fname+std::string("' unexpectedly."));
}
//------------------------------------------------------------------------------------------------------
// Function to read AND parse the next content line
// * check that we really get N values of type T from the current line
template<typename T, int N>
typename std::vector<T>  ParseNextContentLine( std::istream& input, int& lineno, const std::string& fname ) {
  std::string line = ReadNextContentLine(input,lineno,fname);
  return parse_N_values_of_type_T<T,N>(line,lineno,fname);
}
//------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
GateSourceTPSPencilBeam::GateSourceTPSPencilBeam(G4String name ):GateVSource( name ), mPencilBeam(NULL), mDistriGeneral(NULL)
{
  //Particle Type
  strcpy(mParticleType,"proton");
  //Particle Properties if GenericIon
  mIsGenericIon=false;
  //others
  mDistanceSMXToIsocenter=1000;
  mDistanceSMYToIsocenter=1000;
  mDistanceSourcePatient=500;
  pMessenger = new GateSourceTPSPencilBeamMessenger(this);
  mSortedSpotGenerationFlag=false; // try to be backwards compatible
  mSigmaEnergyInMeVFlag = false; // compatibility with wiki documentation, not with 7.2 code
  mTestFlag=false;
  mCurrentParticleNumber=0;
  mCurrentSpot=-1;
  mFlatGenerationFlag=false;
  mIsASourceDescriptionFile=false;
  mSpotIntensityAsNbIons=false;
  mIsInitialized=false;
  mConvergentSourceXTheta=false;
  mConvergentSourceYPhi=false;
  mSelectedLayerID = -1; // all layer selected by default
  mSelectedSpot = -1; // all spots selected by default
  mTotalNbIons = 0.;
}
//------------------------------------------------------------------------------------------------------
GateSourceTPSPencilBeam::~GateSourceTPSPencilBeam() {
  delete pMessenger;  // commented due to segfault
  //FIXME segfault when uncommented
//  delete mPencilBeam;
//  delete mPDF;
//  delete mDistriGeneral;
}
//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::GenerateVertex( G4Event *aEvent ) {
  bool need_pencilbeam_config = false;
  if (!mIsInitialized) {
    GateMessage("Beam", 1, "[TPSPencilBeam] Starting..." << Gateendl );
    // get GATE random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();
    // the "false" means -> do not create messenger (memory gain)
    mPencilBeam = new GateSourcePencilBeam("PencilBeam", false);

    //---------INITIALIZATION - START----------------------
    mIsInitialized = true;
    double NbIons = 0.; // actually: spot weight

    if (mSelectedLayerID != -1 && mSelectedLayerID%2 == 1){
      GateError("Invalid LayerID selected! Select the first ControlPointIndex of a pair (an even number).");
    }

    if (mIsASourceDescriptionFile) {
      LoadClinicalBeamProperties();
      GateMessage("Beam", 0, "[TPSPencilBeam] Source description file successfully loaded." << Gateendl );
    } else {
      GateError("No clinical beam loaded !");
    }

    std::ifstream inFile(mPlan);
    if (! inFile) {
      GateError("Cannot open Treatment plan file!");
    }

    // integrating the plan description file data
    try {
      int spotcounter = -1; //we count spots during init, to be able to match /selectSpot correctly.
      int lineno = 0;
      std::string dummy_PlanName = ReadNextContentLine(inFile,lineno,mPlan);
      int dummy_NbOfFractions = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0]; // not used
      std::string dummy_FractionID = ParseNextContentLine<std::string,1>(inFile,lineno,mPlan)[0]; // not used
      if ( dummy_NbOfFractions != 1){
        GateMessage("Beam",0,"WARNING: nb of fractions is assumed to be 1, but plan file says: " << dummy_NbOfFractions << " and fractionID=" << dummy_FractionID << Gateendl);
      }
      int NbFields = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
      for (int f = 0; f < NbFields; f++) {
        // field IDs, not used
        int dummy_fieldID = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
        GateMessage("Beam",4,"Field ID " << dummy_fieldID << Gateendl );
      }
      double TotalMeterSet = ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0];
      int nrejected = 0; // number of spots rejected based on layer/spot selection configuration      
      if (mTestFlag) {
            GateMessage( "Beam", 0, "TESTREAD NbFields " << NbFields << Gateendl );
            GateMessage( "Beam", 0, "TESTREAD TotalMeterSet " << TotalMeterSet << Gateendl );
         }      
      for (int f = 0; f < NbFields; f++) {
        int FieldID = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
        double MeterSetWeight = ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0];
        GateMessage("Beam",4,"TODO: check that total MSW for this field is indeed " << MeterSetWeight << Gateendl );
        double GantryAngle = deg2rad(ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0]);
        double CouchAngle = deg2rad(ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0]);
        std::vector<double> IsocenterPosition = ParseNextContentLine<double,3>(inFile,lineno,mPlan);
        int NbOfLayers = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
		if (mTestFlag) {
            GateMessage( "Beam", 0, "TESTREAD FieldID " << FieldID << Gateendl );
            GateMessage( "Beam", 0, "TESTREAD GantryAngle " << GantryAngle << Gateendl );
            GateMessage( "Beam", 0, "TESTREAD CouchAngle " << CouchAngle << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD IsocenterPosition " << IsocenterPosition[0] << " " << IsocenterPosition[1] << " " << IsocenterPosition[2] << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD NbOfLayers " << NbOfLayers << Gateendl );
         }
        for (int j = 0; j < NbOfLayers; j++) {
          int currentLayerID = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
          std::string dummy_spotID = ParseNextContentLine<std::string,1>(inFile,lineno,mPlan)[0];
          GateMessage("Beam",4,"spot ID " << dummy_spotID << Gateendl );
          int dummy_cumulative_msw = ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0];
          GateMessage("Beam",4,"cumulative MSW = " << dummy_cumulative_msw << Gateendl );
          double energy = ParseNextContentLine<double,1>(inFile,lineno,mPlan)[0];
          int NbOfSpots = ParseNextContentLine<int,1>(inFile,lineno,mPlan)[0];
          if (mTestFlag) {
            GateMessage( "Beam", 0, "TESTREAD Layers No. " << j << Gateendl );
            GateMessage( "Beam", 0, "TESTREAD NbOfSpots " << NbOfSpots << Gateendl );
          }
          for (int k = 0; k < NbOfSpots; k++) {
            std::vector<double> SpotParameters = ParseNextContentLine<double,3>(inFile,lineno,mPlan);
            if (mTestFlag) {
              GateMessage( "Beam", 1, "TESTREAD Spot No. " << k << " (for this layer)   parameters: "
                                      << SpotParameters[0] << " "
                                      << SpotParameters[1] << " "
                                      << SpotParameters[2] << Gateendl);
            }

            bool allowedField = true;
            // if mNotAllowedFields was set, then check if FieldID was NotAllowed
            if (!mNotAllowedFields.empty()) if ( std::count(mNotAllowedFields.begin(), mNotAllowedFields.end(), FieldID) >  0 ) allowedField = false;
            // if mAllowedFields was set, then check if FieldID was not Allowed.
            if (!mAllowedFields.empty()   ) if ( std::count(mAllowedFields.begin()   , mAllowedFields.end()   , FieldID) == 0 ) allowedField = false;

            bool allowedLayer = true;
            if ((mSelectedLayerID != -1) && (currentLayerID != mSelectedLayerID)) allowedLayer = false;

            bool allowedSpot = true;
            //if disallowed field, layer, or empty spot, skip spot
            if (!allowedField || !allowedLayer || SpotParameters[2] == 0) allowedSpot = false;

            // count spots.
            if (allowedSpot){
              spotcounter++;
              // Skip if /selectSpot was set.
              if ((mSelectedSpot != -1) && (spotcounter != mSelectedSpot)) allowedSpot = false;
            }

            if (allowedField && allowedLayer && allowedSpot) { // loading the spots only for allowed fields
              if (mTestFlag) {
                GateMessage( "Beam", 1, "TESTREAD Spot Loaded. No " << spotcounter << " (from all fields)  parameters: " << SpotParameters[0] << " " << SpotParameters[1] << " " << SpotParameters[2] << Gateendl );
              }

              //POSITION
              // To calculate the beam position with a gantry angle
              G4ThreeVector position;
              position[0] = SpotParameters[0] * (mDistanceSMXToIsocenter - mDistanceSourcePatient) / mDistanceSMXToIsocenter;
              position[1] = SpotParameters[1] * (mDistanceSMYToIsocenter - mDistanceSourcePatient) / mDistanceSMYToIsocenter;
              position[2] = mDistanceSourcePatient;
              //correct orientation problem by rotation 90 degrees around x-Axis
              double xCorrection = halfpi; // 90.*TMath::Pi() / 180.;
              position.rotateX(xCorrection);
              //include gantry rotation
              position.rotateZ(GantryAngle);

              if (mTestFlag) {
                GateMessage( "Beam", 1, "TESTREAD Spot Effective source position " << position[0] << " " << position[1] << " " << position[2] << Gateendl );
              }

              //DIRECTION
              // To calculate the 3 required rotation angles to rotate the beam according to the direction set in the TPS
              G4ThreeVector rotation, direction, test;

              rotation[0] = -xCorrection; //270 degrees

              // deltaY in the patient plan
              double y = atan(SpotParameters[1] / mDistanceSMYToIsocenter);
              rotation[0] += y;
              // deltaX in the patient plan
              double x = -atan(SpotParameters[0] / mDistanceSMXToIsocenter);
              double z = 0.;

              rotation[1]=z;
              rotation[2]=x;
              //set gantry angle rotation
              rotation[2] += GantryAngle;

              if (mTestFlag) {
                GateMessage( "Beam", 1, "TESTREAD source rotation " << rotation[0] << " " << rotation[1] << " " << rotation[2] << Gateendl );
              }
              if (mSpotIntensityAsNbIons) {
                NbIons = SpotParameters[2];
              } else {
                NbIons = SpotParameters[2]*GetMonitorCalibration(energy);
              }
              mTotalNbIons += NbIons;
              mSpotEnergy.push_back(energy);
              mSpotWeight.push_back(NbIons);
              mSpotPosition.push_back(position);
              mSpotRotation.push_back(rotation);
              mSpotLayer.push_back(currentLayerID);

            } else if (mTestFlag) {
              ++nrejected;
              GateMessage("Beam",1,"Rejected spot nr " << k << " (for this layer) for energy=" << energy << " MeV, layer " << j << " in field=" << f << " lineno=" << lineno << Gateendl );
            }
          }
        }
      }
      mTotalNumberOfSpots = mSpotWeight.size();
      mTotalNumberOfLayers = mSpotLayer.back();
      GateMessage("Beam", 1, "[TPSPencilBeam] Plan description file \"" << mPlan << "\" loaded: with " << mTotalNumberOfSpots << " spots loaded and " << nrejected << " spots rejected." << Gateendl );
    } catch ( const std::runtime_error& oops ){
      GateError("Something went wrong while parsing plan description file \"" << mPlan << "\": " << Gateendl << oops.what() << Gateendl );
    }
    inFile.close();

    if (mTotalNumberOfSpots == 0) {
      GateError("0 spots have been loaded from the file \"" << mPlan << "\" simulation abort!");
    }
    
	 //PDF
    mPDF = new double[mTotalNumberOfSpots];

    if (mFlatGenerationFlag) {
      GateMessage("Beam", 0, "WARNING [TPSPencilBeam]: flat generation flag is ON (not recommended for patient simulation)" << Gateendl);
    }

    for (int i = 0; i < mTotalNumberOfSpots; i++) {
      // it is strongly adviced to set mFlatGenerationFlag=false for efficiency 
      if (mFlatGenerationFlag) {
        mPDF[i] = 1;
      } else {
        mPDF[i] = mSpotWeight[i];
      }
    }
    mDistriGeneral = new RandGeneral(engine, mPDF, mTotalNumberOfSpots, 0);
    if (mSortedSpotGenerationFlag){
      mNbIonsToGenerate.resize(mTotalNumberOfSpots,0);
      long int ntotal = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries();
      for (long int i = 0; i<ntotal; i++){
        int bin = mTotalNumberOfSpots * mDistriGeneral->fire();
        ++mNbIonsToGenerate[bin];
      }
      for (int i = 0; i < mTotalNumberOfSpots; i++) {
        GateMessage("Beam", 3, "[TPSPencilBeam] bin " << std::setw(5) << i << ": spotweight=" << std::setw(8) << mPDF[i] << ", Ngen=" << mNbIonsToGenerate[i] << Gateendl );
      }
      mCurrentSpot = 0;
    }

  //Correlation Position/Direction
  //this parameter is not spot or energy dependent and is therefore once for all at the end of the initialization phase.
      if (mConvergentSourceXTheta) {
    mPencilBeam->SetEllipseXThetaRotationNorm("positive");   // convergent beam    
  	 } else {
    	mPencilBeam->SetEllipseXThetaRotationNorm("negative");   // divergent beam
  	   }
  if (mConvergentSourceYPhi) {
	 mPencilBeam->SetEllipseYPhiRotationNorm("positive"); // convergent beam
  	 } else {  
  	   mPencilBeam->SetEllipseYPhiRotationNorm("negative"); // divergent beam
  		}
	// pencil beam configuration
   need_pencilbeam_config = true;
   
   GateMessage("Beam", 0, "[TPSPencilBeam] Plan description file \"" << mPlan << "\" successfully loaded."<< Gateendl );
  }
  //---------INITIALIZATION - END-----------------------

  //---------GENERATION - START-----------------------
  if (mSortedSpotGenerationFlag){
    while ( (mCurrentSpot<mTotalNumberOfSpots) && (mNbIonsToGenerate[mCurrentSpot] <= 0) ){
      GateMessage("Beam", 4, "[TPSPencilBeam] spot " << mCurrentSpot << " has no ions left to generate." << Gateendl );
      mCurrentSpot++;
      mCurrentLayer = mSpotLayer[mCurrentSpot];
      need_pencilbeam_config = true;
    }
    if ( mCurrentSpot>=mTotalNumberOfSpots ){
      GateError("Too many primary vertex requests!");
    }
  } else {
    int nextspot = mTotalNumberOfSpots * mDistriGeneral->fire();
    need_pencilbeam_config = (nextspot!=mCurrentSpot);
    GateMessage("Beam", 5, "[TPSPencilBeam] hopping from spot " << mCurrentSpot << " to spot " << nextspot << Gateendl );
    mCurrentSpot = nextspot;
    mCurrentLayer = mSpotLayer[mCurrentSpot];
  }
  if ( need_pencilbeam_config ){
    GateMessage("Beam", 5, "[TPSPencilBeam] mCurrentSpot = " << mCurrentSpot << Gateendl );
    ConfigurePencilBeam();
  }
  mPencilBeam->GenerateVertex(aEvent);
  if (mSortedSpotGenerationFlag){
    --mNbIonsToGenerate[mCurrentSpot];
  }
}
//---------GENERATION - END-----------------------

//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::ConfigurePencilBeam() {
  double energy = mSpotEnergy[mCurrentSpot];
    //Particle Type
    mPencilBeam->SetParticleType(mParticleType);
  if (mIsGenericIon==true){
    //Particle Properties If GenericIon
    mPencilBeam->SetIonParameter(mParticleParameters);
  }
  //Energy
  mPencilBeam->SetEnergy(GetEnergy(energy));
  if ( mSigmaEnergyInMeVFlag ){
    mPencilBeam->SetEnergy(GetEnergy(energy));
    mPencilBeam->SetSigmaEnergy(GetSigmaEnergy(energy));
  } else {
    double source_energy = GetEnergy(energy);
    mPencilBeam->SetEnergy(source_energy);
    mPencilBeam->SetSigmaEnergy(GetSigmaEnergy(energy)*source_energy/100.);
  }
  //Weight
  if (mFlatGenerationFlag) {
    mPencilBeam->SetWeight(mSpotWeight[mCurrentSpot]);
  } else {
    mPencilBeam->SetWeight(1.);
  }
  //Position
  mPencilBeam->SetPosition(mSpotPosition[mCurrentSpot]);
  mPencilBeam->SetSigmaX(GetSigmaX(energy));
  mPencilBeam->SetSigmaY(GetSigmaY(energy));
  //Direction
  mPencilBeam->SetSigmaTheta(GetSigmaTheta(energy));
  mPencilBeam->SetEllipseXThetaArea(GetEllipseXThetaArea(energy));
  mPencilBeam->SetSigmaPhi(GetSigmaPhi(energy));
  mPencilBeam->SetEllipseYPhiArea(GetEllipseYPhiArea(energy));
  mPencilBeam->SetRotation(mSpotRotation[mCurrentSpot]);
  //Correlation Position/Direction
  //this parameter is not spot or energy dependent and is therefore once for all at the end of the initialization phase.

  mPencilBeam->SetTestFlag(mTestFlag);
  if (mTestFlag) {
    GateMessage("Beam", 0, "Configuration of spot (ID) No. " << mCurrentSpot << " (out of " << mTotalNumberOfSpots << ")" << Gateendl);
    GateMessage("Beam", 0, "Energy\t" << energy << Gateendl);
    GateMessage("Beam", 0, "Spot metersetweight\t" << mSpotWeight[mCurrentSpot] << Gateendl);
    GateMessage("Beam", 0, "Total Spot metersetweight\t" << mTotalNbIons << Gateendl);
    GateMessage("Beam", 0, "SetEnergy\t" << GetEnergy(energy) << Gateendl);
    GateMessage("Beam", 0, "SetSigmaEnergy\t" << GetSigmaEnergy(energy) << Gateendl);
    GateMessage("Beam", 0, "SetSigmaX\t" << GetSigmaX(energy) << Gateendl);
    GateMessage("Beam", 0, "SetSigmaY\t" << GetSigmaY(energy) << Gateendl);
    GateMessage("Beam", 0, "SetSigmaTheta\t" << GetSigmaTheta(energy) << Gateendl);
    GateMessage("Beam", 0, "SetSigmaPhi\t" << GetSigmaPhi(energy) << Gateendl);
    GateMessage("Beam", 0, "SetEllipseXThetaArea\t" << GetEllipseXThetaArea(energy) << Gateendl);
    GateMessage("Beam", 0, "SetEllipseYPhiArea\t" << GetEllipseYPhiArea(energy) << Gateendl);
  }
}

//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::LoadClinicalBeamProperties() {

  std::string oneline;
  int PolOrder;
  int lineno=0;
  double MyVal;

  std::ifstream inFile(mSourceDescriptionFile);
  if (! inFile) {
    GateError("Cannot open source description file!");
  }
  // DSP, SMX, SMY
  mDistanceSourcePatient=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mDistanceSMXToIsocenter=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mDistanceSMYToIsocenter=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
  //Energy
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mEnergy.push_back(PolOrder);
  for (int i=0; i<=PolOrder; i++) {
      MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
      mEnergy.push_back(MyVal);
  }
  for (int i=0; i<4; i++) std::getline(inFile,oneline);
  // Energy Spread
  if (oneline == "MeV"){
    mSigmaEnergyInMeVFlag = true;
    GateMessage("Beam",0,"source description file specifies energy spread in MeV" << Gateendl);
    GateMessage("Beam",0,"(This overrides whatever you configured for the 'setSigmaEnergyInMeVFlag' in the configuration of TPSPencilBeam.)" << Gateendl);
    std::getline(inFile,oneline);
  } else if ( (oneline == "PERCENT") || (oneline == "percent") || (oneline == "%") ){
    mSigmaEnergyInMeVFlag = false;
    GateMessage("Beam",0,"source description file specifies energy spread in PERCENT (%)" << Gateendl);
    GateMessage("Beam",0,"(This overrides whatever you configured for the 'setSigmaEnergyInMeVFlag' in the configuration of TPSPencilBeam.)" << Gateendl);
    std::getline(inFile,oneline);
  }
  PolOrder=atoi(oneline.c_str());
  mEnergySpread.push_back(PolOrder);
  for (int i=0; i<=PolOrder; i++) {
    MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
    mEnergySpread.push_back(MyVal);
  }
  //X
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mX.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mX.push_back(MyVal);
    }
  //Theta
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mTheta.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mTheta.push_back(MyVal);
    }
  //Y
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mY.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mY.push_back(MyVal);
    }
  //Phi
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mPhi.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mPhi.push_back(MyVal);
    }
  //XThetaEmittance
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mXThetaEmittance.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mXThetaEmittance.push_back(MyVal);
    }
  //YPhiEmittance
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mYPhiEmittance.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mYPhiEmittance.push_back(MyVal);
    }
    
  if(!mSpotIntensityAsNbIons){
  //MonitorCalibration
  PolOrder=ParseNextContentLine<int,1>(inFile,lineno,mSourceDescriptionFile)[0];
  mMonitorCalibration.push_back(PolOrder);
    for (int i=0; i<=PolOrder; i++) {
        MyVal=ParseNextContentLine<double,1>(inFile,lineno,mSourceDescriptionFile)[0];
        mMonitorCalibration.push_back(MyVal);
    }
  }
  //TestFlag
  if (mTestFlag) {
    GateMessage("Beam",0,"TESTREAD DSP "<<mDistanceSourcePatient<< Gateendl);
    GateMessage("Beam",0,"TESTREAD SMX "<<mDistanceSMXToIsocenter<< Gateendl);
    GateMessage("Beam",0,"TESTREAD SMY "<<mDistanceSMYToIsocenter<< Gateendl);
    for (unsigned int i=0; i<mEnergy.size(); i++) GateMessage("Beam",0,"TESTREAD mEnergy\t"<<mEnergy[i]<< Gateendl);
    for (unsigned int i=0; i<mEnergySpread.size(); i++) GateMessage("Beam",0,"TESTREAD mEnergySpread\t"<<mEnergySpread[i]<< Gateendl);
    for (unsigned int i=0; i<mX.size(); i++) GateMessage("Beam",0,"TESTREAD mX\t"<<mX[i]<< Gateendl);
    for (unsigned int i=0; i<mTheta.size(); i++) GateMessage("Beam",0,"TESTREAD mTheta\t"<<mTheta[i]<< Gateendl);
    for (unsigned int i=0; i<mY.size(); i++) GateMessage("Beam",0,"TESTREAD mY\t"<<mY[i]<< Gateendl);
    for (unsigned int i=0; i<mPhi.size(); i++) GateMessage("Beam",0,"TESTREAD mPhi\t"<<mPhi[i]<< Gateendl);
    for (unsigned int i=0; i<mXThetaEmittance.size(); i++) GateMessage("Beam",0,"TESTREAD mXThetaEmittance\t"<<mXThetaEmittance[i]<< Gateendl);
    for (unsigned int i=0; i<mYPhiEmittance.size(); i++) GateMessage("Beam",0,"TESTREAD mYPhiEmittance\t"<<mYPhiEmittance[i]<< Gateendl);
    for (unsigned int i=0; i<mMonitorCalibration.size(); i++) GateMessage("Beam",0,"TESTREAD mMonitorCalibration\t"<<mMonitorCalibration[i]<< Gateendl);
  }
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEnergy(double energy) {
  double val=0;
  for (int i=0; i<=mEnergy[0]; i++) {
    val+=mEnergy[i+1]*pow(energy,mEnergy[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaEnergy(double energy) {
  double val=0;
  for (int i=0; i<=mEnergySpread[0]; i++) {
    val+=mEnergySpread[i+1]*pow(energy,mEnergySpread[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaX(double energy) {
  double val=0;
  for (int i=0; i<=mX[0]; i++) {
    val+=mX[i+1]*pow(energy,mX[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaY(double energy) {
  double val=0;
  for (int i=0; i<=mY[0]; i++) {
    val+=mY[i+1]*pow(energy,mY[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaTheta(double energy) {
  double val=0;
  for (int i=0; i<=mTheta[0]; i++) {
    val+=mTheta[i+1]*pow(energy,mTheta[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaPhi(double energy) {
  double val=0;
  for (int i=0; i<=mPhi[0]; i++) {
    val+=mPhi[i+1]*pow(energy,mPhi[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEllipseXThetaArea(double energy) {
  double val=0;
  for (int i=0; i<=mXThetaEmittance[0]; i++) {
    val+=mXThetaEmittance[i+1]*pow(energy,mXThetaEmittance[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEllipseYPhiArea(double energy) {
  double val=0;
  for (int i=0; i<=mYPhiEmittance[0]; i++) {
    val+=mYPhiEmittance[i+1]*pow(energy,mYPhiEmittance[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetMonitorCalibration(double energy) {
  double val=0;
  for (int i=0; i<=mMonitorCalibration[0]; i++) {
    val+=mMonitorCalibration[i+1]*pow(energy,mMonitorCalibration[0]-i);
  }
  return val;
}
//------------------------------------------------------------------------------------------------------
G4int GateSourceTPSPencilBeam::GeneratePrimaries( G4Event* event ) {
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << Gateendl);
  G4int numVertices = 0;
  GenerateVertex( event );
  numVertices++;
  return numVertices;
}
//------------------------------------------------------------------------------------------------------
void ReadLineTo3Doubles(double *toto, const std::string &data) {
  std::istringstream iss(data);
  std::string token;
  for (int j=0; j<3; j++) {
    getline(iss, token, ' ');
    toto[j]=atof(token.c_str());
  }
}
// vim: ai sw=2 ts=2 et