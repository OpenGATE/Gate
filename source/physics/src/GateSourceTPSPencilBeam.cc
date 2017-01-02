/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//=======================================================
//  Created by Loïc Grevillot
//  General definition of the class
// This class allows for simulating treatment plans for Pencil Beam Scanning applications.
// The source need 2 inputs: a beam model of the system and a treatment plans
// It will simulate each single pencil beam of the treatment plan using the GateSourcePencilBeam class.
//=======================================================

//Modified by Hermann Fuchs
//Medical University Vienna
//Added missing Couch Angle
//Corrected angle calculation to be conform with DICOM standards
//Corrected energy distribution to conform to expected behaviour and publication

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
GateSourceTPSPencilBeam::GateSourceTPSPencilBeam(G4String name ):GateVSource( name ), mPencilBeam(NULL), mDistriGeneral(NULL)
{

  strcpy(mParticleType,"proton");
  mDistanceSMXToIsocenter=1000;
  mDistanceSMYToIsocenter=1000;
  mDistanceSourcePatient=500;
  pMessenger = new GateSourceTPSPencilBeamMessenger(this);
  mOldStyleFlag=false;
  mTestFlag=false;
  mCurrentParticleNumber=0;
  mCurrentSpot=0;
  mFlatGenerationFlag=false;
  mIsASourceDescriptionFile=false;
  mSpotIntensityAsNbProtons=false;
  mIsInitialized=false;
  mConvergentSource=false;
  mSelectedLayerID = -1; // all layer selected by default
  mSelectedSpot = -1; // all spots selected by default
  mTotalNbProtons = 0.;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
GateSourceTPSPencilBeam::~GateSourceTPSPencilBeam() {
  delete pMessenger;  // commented due to segfault
  //  for (int i=0; i<mPencilBeams.size(); i++)  { delete mPencilBeams[i]; }
  //FIXME segfault when uncommented
  //if (mDistriGeneral) delete mDistriGeneral;
  // maybe we should delete mPencilBeam, maybe not...
}
//------------------------------------------------------------------------------------------------------

void GateSourceTPSPencilBeam::GenerateVertex( G4Event *aEvent ) {
  if (mOldStyleFlag) {
    this->OldGenerateVertex(aEvent);
  } else {
    this->NewGenerateVertex(aEvent);
  }
}

//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::OldGenerateVertex( G4Event *aEvent ) {

  if (!mIsInitialized) {
    // get GATE random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();

    //---------INITIALIZATION - START----------------------
    mIsInitialized = true;

    std::string oneline;
    int NbFields, FieldID, TotalMeterSet, NbOfLayers;
    double GantryAngle;
    double CouchAngle;
    double IsocenterPosition[3];
    double NbProtons;
    bool again = true;
    //again is a check to skip the second controlpoint of each controlpointindex pair.
    //an extra check is needed when the selectLayerID switch is used.
    if (mSelectedLayerID != -1 && mSelectedLayerID%2 == 1){
      GateError("Invalid LayerID selected! Select the first ControlPointIndex of a pair (an even number).");
    }

    if (mIsASourceDescriptionFile) {
      LoadClinicalBeamProperties();
      GateMessage("Beam", 1, "[TPSPencilBeam] Source description file successfully loaded." << Gateendl);
    } else {
      GateError("No clinical beam loaded !");
    }

    std::ifstream inFile(mPlan);
    if (! inFile) {
      GateError("Cannot open Treatment plan file!");
    }

    // integrating the plan description file data
    while (inFile && again) {
      int spotcounter = -1; //we count spots during init, to be able to match /selectSpot correctly.

      for (int i = 0; i < 9; i++) std::getline(inFile,oneline);
      NbFields = atoi(oneline.c_str());
      for (int i = 0; i < 2 * NbFields; i++) std::getline(inFile,oneline);
      for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
      TotalMeterSet = atoi(oneline.c_str());

      for (int f = 0; f < NbFields; f++) {
        for (int i = 0; i < 4; i++) std::getline(inFile,oneline);
        FieldID = atoi(oneline.c_str());

        for (int i = 0; i < 4; i++) std::getline(inFile,oneline);
        GantryAngle = deg2rad(atof(oneline.c_str()));

        //MISSING COUCH ANGLE inserted
        for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
        CouchAngle = deg2rad(atof(oneline.c_str()));

        for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
        ReadLineTo3Doubles(IsocenterPosition, oneline);
        for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
        NbOfLayers = atoi(oneline.c_str());
        for (int i = 0; i < 2; i++) std::getline(inFile,oneline);

        for (int j = 0; j < NbOfLayers; j++) {
          for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
          int currentLayerID = atoi(oneline.c_str()); // ControlPointIndex

          for (int i = 0; i < 6; i++) std::getline(inFile,oneline);
          double energy = atof(oneline.c_str());

          for (int i = 0; i < 2; i++) std::getline(inFile,oneline);
          int NbOfSpots = atof(oneline.c_str());
          for (int i = 0; i < 1; i++) std::getline(inFile,oneline);

          if (mTestFlag) {
            G4cout << "TESTREAD NbFields " << NbFields << Gateendl;
            G4cout << "TESTREAD TotalMeterSet " << TotalMeterSet << Gateendl;
            G4cout << "TESTREAD FieldID " << FieldID << Gateendl;
            G4cout << "TESTREAD GantryAngle " << GantryAngle << Gateendl;
            G4cout << "TESTREAD CouchAngle " << CouchAngle << Gateendl;
            G4cout << "TESTREAD Layers N° " << j << Gateendl;
            G4cout << "TESTREAD NbOfSpots " << NbOfSpots << Gateendl;
          }
          for (int k = 0; k < NbOfSpots; k++) {
            std::getline(inFile,oneline);
            double SpotParameters[3];
            ReadLineTo3Doubles(SpotParameters, oneline);

            if (mTestFlag) {
              G4cout << "TESTREAD Spot N° " << k << "    parameters: " << SpotParameters[0] << " " << SpotParameters[1] << " " << SpotParameters[2] << Gateendl;
            }

            //POSITION
            // To calculate the beam position with a gantry angle
            G4ThreeVector position;
            position[0] = SpotParameters[0] * (mDistanceSMXToIsocenter - mDistanceSourcePatient) / mDistanceSMXToIsocenter;
            position[1] = SpotParameters[1] * (mDistanceSMYToIsocenter - mDistanceSourcePatient) / mDistanceSMYToIsocenter;
            position[2] = mDistanceSourcePatient;
            //            position[0]=SpotParameters[0]*(mDistanceSMXToIsocenter-mDistanceSourcePatient)/mDistanceSMXToIsocenter;
            //        position[1]=SpotParameters[1]*(mDistanceSMYToIsocenter-mDistanceSourcePatient)/mDistanceSMYToIsocenter;
            //        position[2]=mDistanceSourcePatient;
            //correct orientation problem by rotation 90 degrees around x-Axis
            double xCorrection = halfpi; // 90.*TMath::Pi() / 180.;
            //            position.rotateX(xCorrection-CouchAngle);
            position.rotateX(xCorrection - CouchAngle);
            //if (GantryAngle!=0)//orig
            //            position.rotateY(GantryAngle);//orig
            //position.rotateY(GantryAngle);

            //include couch rotation
            //if (CouchAngle!=0)
            //            position.rotateY(CouchAngle);
            //include gantry rotation
            //if (GantryAngle!=0)
            position.rotateZ(GantryAngle);
            //            position.rotateX(-CouchAngle);

            if (mTestFlag) {
              G4cout << "TESTREAD Spot Effective source position " << position[0] << " " << position[1] << " " << position[2] << Gateendl;
              G4cout << "TESTREAD IsocenterPosition " << IsocenterPosition[0] << " " << IsocenterPosition[1] << " " << IsocenterPosition[2] << Gateendl;
              G4cout << "TESTREAD NbOfLayers " << NbOfLayers << Gateendl;
            }

            //DIRECTION
            // To calculate the 3 required rotation angles to rotate the beam according to the direction set in the TPS
            G4ThreeVector rotation, direction, test;


            // GantryAngle at 0 (Default)
            //            //ORIGINAL BLOCK
            //            rotation[0]=TMath::Pi();
            //        // deltaY in the patient plan
            //        rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
            //        // deltaX in the patient plan
            //        rotation[1]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
            //        // no gantry head rotation
            //        rotation[2]=0;
            //        //set gantry angle rotation
            //            rotation[1]+=GantryAngle;


            //                        // GantryAngle at 0 (Default)
            //            //rotation[0]=TMath::Pi()+xCorrection;//270 degrees
            //            rotation[0]=-xCorrection;//270 degrees
            //
            //        // deltaY in the patient plan
            //        rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
            //        // deltaX in the patient plan
            //        rotation[2]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
            //        // no gantry head rotation
            //        rotation[1]=0.;
            //        //set gantry angle rotation
            //        rotation[2]+=GantryAngle;//+CouchAngle;
            ////            rotation[2]+=CouchAngle;
            ////            rotation[0]+=-CouchAngle;//-couchAngle


            // GantryAngle at 0 (Default)
            //rotation[0]=TMath::Pi()+xCorrection;//270 degrees
            rotation[0] = -xCorrection; //270 degrees

            // deltaY in the patient plan
            double y = atan(SpotParameters[1] / mDistanceSMYToIsocenter);
            rotation[0] += y;
            // deltaX in the patient plan
            double x = -atan(SpotParameters[0] / mDistanceSMXToIsocenter);
            double z = 0.;
            rotation[1] = sin(CouchAngle) * (x) + cos(CouchAngle) * z;
            // no gantry head rotation
            rotation[2] = cos(CouchAngle) * (x) + sin(CouchAngle) * z;
            //set gantry angle rotation
            rotation[2] += GantryAngle; //+CouchAngle;
            //            rotation[2]+=CouchAngle;
            rotation[0] += -CouchAngle; //-couchAngle


            //            rotation[0]=0;//TMath::Pi();
            //        // deltaY in the patient plan
            //        rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
            //        // deltaX in the patient plan
            //        rotation[2]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
            //        // no gantry head rotation
            //        rotation[1]=0;
            //        //set gantry angle rotation
            //            //rotation[1]+=GantryAngle;
            //            rotation[1]-=GantryAngle;
            //            rotation[2]+=0.;

            //            G4cout<<"TESTREAD Spot Effective source position "<<position[0]<<" "<<position[1]<<" "<<position[2]<< Gateendl;
            //            G4cout<<"TESTREAD source rotation "<<rotation[0]<<" "<<rotation[1]<<" "<<rotation[2]<< Gateendl;
            //            G4cout<<"TESTREAD couch angle "<<CouchAngle<< Gateendl;
            //            G4cout<<"TESTREAD gantry angle "<<GantryAngle<< Gateendl;
            //            G4cout<< Gateendl;
            if (mTestFlag) {
              G4cout << "TESTREAD source rotation " << rotation[0] << " " << rotation[1] << " " << rotation[2] << Gateendl;
            }


            // Brent 2014-02-19: This check is in an inner loop, but with good reason: we're in the parsing stage.
            // Rewrote to work also with AllowedFields.
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
                G4cout << "TESTREAD Spot Loaded. N° " << spotcounter << "   parameters: " << SpotParameters[0] << " " << SpotParameters[1] << " " << SpotParameters[2] << Gateendl;
              }
              // the false mean -> do not create messenger (memory gain)
              GateSourcePencilBeam *Pencil = new GateSourcePencilBeam("PencilBeam", false);

              //Particle Type
              Pencil->SetParticleType(mParticleType);
              //Energy
              Pencil->SetEnergy(GetEnergy(energy));
              Pencil->SetSigmaEnergy(GetSigmaEnergy(energy));

              //cerr << "Brent " << GetSigmaEnergy(energy) << " en " << GetEnergy(energy) <<endl;

              //changed because obiously incorrect.
              //Pencil->SetSigmaEnergy(GetSigmaEnergy(energy)*GetEnergy(energy)/100.);
              //Weight

              if (mSpotIntensityAsNbProtons) {
                NbProtons = SpotParameters[2];
              } else {
                NbProtons = ConvertMuToProtons(SpotParameters[2], GetEnergy(energy));
              }

              Pencil->SetWeight(NbProtons);
              //G4cout<<"Nb of MU = "<<SpotParameters[2]<<", beam energy = "<<energy<<" MeV, corresponding to "<<NbProtons<<" protons.\n";
              //Position
              Pencil->SetPosition(position);
              Pencil->SetSigmaX(GetSigmaX(energy));
              Pencil->SetSigmaY(GetSigmaY(energy));
              //Direction
              Pencil->SetSigmaTheta(GetSigmaTheta(energy));
              Pencil->SetEllipseXThetaArea(GetEllipseXThetaArea(energy));
              Pencil->SetSigmaPhi(GetSigmaPhi(energy));
              Pencil->SetEllipseYPhiArea(GetEllipseYPhiArea(energy));
              Pencil->SetRotation(rotation);
              
              mSpotLayer.push_back(currentLayerID);

              //Correlation Position/Direction
              if (mConvergentSource) {
                Pencil->SetEllipseXThetaRotationNorm("positive");   // convergent beam
                Pencil->SetEllipseYPhiRotationNorm("positive"); // convergent beam
              } else {
                Pencil->SetEllipseXThetaRotationNorm("negative");   // divergent beam
                Pencil->SetEllipseYPhiRotationNorm("negative"); // divergent beam
              }
              Pencil->SetTestFlag(mTestFlag);
              //new pencil added
              mPencilBeams.push_back(Pencil);

              if (mTestFlag) {
                G4cout << "Energy\t" << energy << Gateendl;
                G4cout << "SetEnergy\t" << GetEnergy(energy) << Gateendl;
                G4cout << "SetSigmaEnergy\t" << GetSigmaEnergy(energy) << Gateendl;
                G4cout << "SetSigmaX\t" << GetSigmaX(energy) << Gateendl;
                G4cout << "SetSigmaY\t" << GetSigmaY(energy) << Gateendl;
                G4cout << "SetSigmaTheta\t" << GetSigmaTheta(energy) << Gateendl;
                G4cout << "SetSigmaPhi\t" << GetSigmaPhi(energy) << Gateendl;
                G4cout << "SetEllipseXThetaArea\t" << GetEllipseXThetaArea(energy) << Gateendl;
                G4cout << "SetEllipseYPhiArea\t" << GetEllipseYPhiArea(energy) << Gateendl;
              }
            }
          }
        }
      }
      again = false;
      GateMessage("Beam", 1, "[TPSPencilBeam] Plan description file successfully loaded." << Gateendl);
    }
    inFile.close();

    mTotalNumberOfSpots = mPencilBeams.size();
    mTotalNumberOfLayers = mSpotLayer.back();
    if (mTotalNumberOfSpots == 0) {
      GateError("0 spots have been loaded from the file \"" << mPlan << "\" simulation abort!");
    }

    GateMessage("Beam", 1, "[TPSPencilBeam] Starting particle generation:  "
                << mTotalNumberOfSpots << " spots loaded." << Gateendl );
    mPDF = new double[mTotalNumberOfSpots];
    for (int i = 0; i < mTotalNumberOfSpots; i++) {
      // it is strongly adviced to set mFlatGenerationFlag=false
      // a few test demonstrated a lot more efficiency for "real field like" simulation in patients.
      if (mFlatGenerationFlag) {
        mPDF[i] = 1;
      } else {
        mPDF[i] = mPencilBeams[i]->GetWeight();
        mPencilBeams[i]->SetWeight(1);
      }
    }
    mDistriGeneral = new RandGeneral(engine, mPDF, mTotalNumberOfSpots, 0);

    //---------INITIALIZATION - END-----------------------
  }
  //---------OLD GENERATION - START-----------------------
  int bin = mTotalNumberOfSpots * mDistriGeneral->fire();
  mCurrentSpot = bin;
  mCurrentLayer = mSpotLayer[mCurrentSpot];
  mPencilBeams[bin]->GenerateVertex(aEvent);
}
//---------OLD GENERATION - END-----------------------

template<typename T, int N>
typename std::vector<T> parse_N_items_of_type_T(std::string line,int lineno, const std::string& fname){
    std::istringstream sin(line);
    typename std::istream_iterator<T> eos;
    typename std::istream_iterator<T> isd(sin);
    typename std::vector<T> vecT(N);
    typename std::vector<T>::iterator endT = std::copy(isd,eos,vecT.begin());
    size_t nread = endT - vecT.begin();
    if (nread != N){
        std::ostringstream errMsg;
        errMsg << "wrong number of items ("
               << nread << ") on line " << lineno << " of " << fname
               << "; expected " << N << " item(s) of type " << typeid(T).name()<< std::endl;
        throw std::runtime_error(errMsg.str());
    }
    return vecT;
}


// Function to read the next content line
// * skip all comment lines (lines string with a '#')
// * skip empty
// * check that we really get N items of type T from the current line
// * throw exception with informative error message in case of trouble
template<typename T, int N>
typename std::vector<T>  ReadNextContentLine( std::istream& input, int& lineno, const std::string& fname ) {
  while ( input ){
    std::string line;
    std::getline(input,line);
    ++lineno;
    if (line.empty()) continue;
    if (line[0]=='#') continue;
    return parse_N_items_of_type_T<T,N>(line,lineno,fname);
  }
  throw std::runtime_error(std::string("reached end of file")+fname+std::string("unexpectedly"));
}


//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::NewGenerateVertex( G4Event *aEvent ) {

  bool need_pencilbeam_config = false;
  if (!mIsInitialized) {
    GateMessage("Beam", 1, "[TPSPencilBeam] Going to try loading the PBS plan description." << Gateendl );
    // get GATE random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();
    // the "false" means -> do not create messenger (memory gain)
    mPencilBeam = new GateSourcePencilBeam("PencilBeam", false);


    //---------INITIALIZATION - START----------------------
    mIsInitialized = true;

    double NbProtons = 0.; // actually: spot weight

    if (mSelectedLayerID != -1 && mSelectedLayerID%2 == 1){
      GateError("Invalid LayerID selected! Select the first ControlPointIndex of a pair (an even number).");
    }

    if (mIsASourceDescriptionFile) {
      LoadClinicalBeamProperties();
      GateMessage("Beam", 1, "[TPSPencilBeam] Source description file successfully loaded." << Gateendl );
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
      std::string dummy_PlanName = ReadNextContentLine<std::string,1>(inFile,lineno,mPlan)[0];
      int dummy_NbOfFractions = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0]; // not used
      std::string dummy_FractionID = ReadNextContentLine<std::string,1>(inFile,lineno,mPlan)[0]; // not used
      if ( dummy_NbOfFractions != 1){
        GateMessage("Beam",0,"WARNING: nb of fractions is assumed to be 1, but plan file says: " << dummy_NbOfFractions << " and fractionID=" << dummy_FractionID << Gateendl);
      }
      int NbFields = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
      for (int f = 0; f < NbFields; f++) {
        // field IDs, not used
        int dummy_fieldID = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
        GateMessage("Beam",4,"Field ID " << dummy_fieldID << Gateendl );
      }
      double TotalMeterSet = ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0];
      int nrejected = 0; // number of spots rejected based on layer/spot selection configuration
      for (int f = 0; f < NbFields; f++) {
        int FieldID = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
        double MeterSetWeight = ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0];
        GateMessage("Beam",4,"TODO: check that total MSW for this field is indeed " << MeterSetWeight << Gateendl );
        double GantryAngle = deg2rad(ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0]);
        double CouchAngle = deg2rad(ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0]);
        std::vector<double> IsocenterPosition = ReadNextContentLine<double,3>(inFile,lineno,mPlan);
        int NbOfLayers = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
        for (int j = 0; j < NbOfLayers; j++) {
          int currentLayerID = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
          std::string dummy_spotID = ReadNextContentLine<std::string,1>(inFile,lineno,mPlan)[0];
          GateMessage("Beam",4,"spot ID " << dummy_spotID << Gateendl );
          int dummy_cumulative_msw = ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0];
          GateMessage("Beam",4,"cumulative MSW = " << dummy_cumulative_msw << Gateendl );
          double energy = ReadNextContentLine<double,1>(inFile,lineno,mPlan)[0];
          int NbOfSpots = ReadNextContentLine<int,1>(inFile,lineno,mPlan)[0];
          if (mTestFlag) {
            GateMessage( "Beam", 1, "TESTREAD NbFields " << NbFields << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD TotalMeterSet " << TotalMeterSet << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD FieldID " << FieldID << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD GantryAngle " << GantryAngle << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD CouchAngle " << CouchAngle << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD Layers No. " << j << Gateendl );
            GateMessage( "Beam", 1, "TESTREAD NbOfSpots " << NbOfSpots << Gateendl );
          }
          for (int k = 0; k < NbOfSpots; k++) {
            std::vector<double> SpotParameters = ReadNextContentLine<double,3>(inFile,lineno,mPlan);
            if (mTestFlag) {
              GateMessage( "Beam", 1, "TESTREAD Spot No. " << k << "    parameters: "
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
                GateMessage( "Beam", 1, "TESTREAD Spot Loaded. No " << spotcounter << "   parameters: " << SpotParameters[0] << " " << SpotParameters[1] << " " << SpotParameters[2] << Gateendl );
              }

              //POSITION
              // To calculate the beam position with a gantry angle
              G4ThreeVector position;
              position[0] = SpotParameters[0] * (mDistanceSMXToIsocenter - mDistanceSourcePatient) / mDistanceSMXToIsocenter;
              position[1] = SpotParameters[1] * (mDistanceSMYToIsocenter - mDistanceSourcePatient) / mDistanceSMYToIsocenter;
              position[2] = mDistanceSourcePatient;
              //correct orientation problem by rotation 90 degrees around x-Axis
              double xCorrection = halfpi; // 90.*TMath::Pi() / 180.;
              position.rotateX(xCorrection - CouchAngle);
              //include gantry rotation
              position.rotateZ(GantryAngle);

              if (mTestFlag) {
                GateMessage( "Beam", 1, "TESTREAD Spot Effective source position " << position[0] << " " << position[1] << " " << position[2] << Gateendl );
                GateMessage( "Beam", 1, "TESTREAD IsocenterPosition " << IsocenterPosition[0] << " " << IsocenterPosition[1] << " " << IsocenterPosition[2] << Gateendl );
                GateMessage( "Beam", 1, "TESTREAD NbOfLayers " << NbOfLayers << Gateendl );
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
              rotation[1] = sin(CouchAngle) * (x) + cos(CouchAngle) * z;
              // no gantry head rotation
              rotation[2] = cos(CouchAngle) * (x) + sin(CouchAngle) * z;
              //set gantry angle rotation
              rotation[2] += GantryAngle; //+CouchAngle;
              rotation[0] += -CouchAngle; //-couchAngle

              if (mTestFlag) {
                GateMessage( "Beam", 1, "TESTREAD source rotation " << rotation[0] << " " << rotation[1] << " " << rotation[2] << Gateendl );
              }

              if (mSpotIntensityAsNbProtons) {
                NbProtons = SpotParameters[2];
              } else {
                NbProtons = ConvertMuToProtons(SpotParameters[2], GetEnergy(energy));
              }
              mTotalNbProtons += NbProtons;
              mSpotEnergy.push_back(energy);
              mSpotWeight.push_back(NbProtons);
              mSpotPosition.push_back(position);
              mSpotRotation.push_back(rotation);
              mSpotLayer.push_back(currentLayerID);

            } else if (mTestFlag) {
              ++nrejected;
              GateMessage("Beam",1,"Rejected spot nr " << k << " for energy=" << energy << " MeV, layer " << j << " in field=" << f << " lineno=" << lineno << Gateendl );
            }
          }
        }
      }
      mTotalNumberOfSpots = mSpotWeight.size();
      mTotalNumberOfLayers = mSpotLayer.back();
      GateMessage("Beam", 1, "[TPSPencilBeam] Plan description file \"" << mPlan << "\" successfully loaded: " << NbFields << " field(s) with a total of " << mTotalNumberOfSpots << " spots, " << nrejected << " spots rejected." << Gateendl );
    } catch ( const std::runtime_error& oops ){
      GateError("Something went wrong while parsing plan description file \"" << mPlan << "\": " << Gateendl << oops.what() << Gateendl );
    }
    inFile.close();

    if (mTotalNumberOfSpots == 0) {
      GateError("0 spots have been loaded from the file \"" << mPlan << "\" simulation abort!");
    }

    mPDF = new double[mTotalNumberOfSpots];
    if (mFlatGenerationFlag) {
      GateMessage("Beam", 0, "WARNING [TPSPencilBeam]: flat generation flag is ON (not recommended for patient simulation)" << Gateendl);
    }
    for (int i = 0; i < mTotalNumberOfSpots; i++) {
      // it is strongly adviced to set mFlatGenerationFlag=false
      // a few test demonstrated a lot more efficiency for "real field like" simulation in patients.
      if (mFlatGenerationFlag) {
        mPDF[i] = 1;
      } else {
        mPDF[i] = mSpotWeight[i];
      }
    }
    mDistriGeneral = new RandGeneral(engine, mPDF, mTotalNumberOfSpots, 0);
    mNbProtonsToGenerate.resize(mTotalNumberOfSpots,0);
    long int ntotal = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries();
    for (long int i = 0; i<ntotal; i++){
      int bin = mTotalNumberOfSpots * mDistriGeneral->fire();
      ++mNbProtonsToGenerate[bin];
    }
    for (int i = 0; i < mTotalNumberOfSpots; i++) {
      GateMessage("Beam", 3, "[TPSPencilBeam] bin " << std::setw(5) << i << ": spotweight=" << std::setw(8) << mPDF[i] << ", Ngen=" << mNbProtonsToGenerate[i] << Gateendl );
    }
    need_pencilbeam_config = true;
    GateMessage("Beam", 1, "[TPSPencilBeam] Plan description file successfully loaded." << Gateendl );

    //---------INITIALIZATION - END-----------------------
  }
  //---------GENERATION - START-----------------------
  while ( (mCurrentSpot<mTotalNumberOfSpots) && (mNbProtonsToGenerate[mCurrentSpot] <= 0) ){
    GateMessage("Beam", 4, "[TPSPencilBeam] spot " << mCurrentSpot << " has no protons left to generate." << Gateendl );
    mCurrentSpot++;
    mCurrentLayer = mSpotLayer[mCurrentSpot];
    need_pencilbeam_config = true;
  }
  if ( mCurrentSpot>=mTotalNumberOfSpots ){
    GateError("Too many primary vertex requests!");
  }
  if ( need_pencilbeam_config ){
    GateMessage("Beam", 4, "[TPSPencilBeam] configuring pencil beam for spot " << mCurrentSpot
        << ", to generate " << mNbProtonsToGenerate[mCurrentSpot] << " protons." << Gateendl );
    ConfigurePencilBeam();
  }
  mPencilBeam->GenerateVertex(aEvent);
  --mNbProtonsToGenerate[mCurrentSpot];
}
//---------GENERATION - END-----------------------

void GateSourceTPSPencilBeam::ConfigurePencilBeam() {
  double energy = mSpotEnergy[mCurrentSpot];
  //Particle Type
  mPencilBeam->SetParticleType(mParticleType);
  //Energy
  mPencilBeam->SetEnergy(GetEnergy(energy));
  mPencilBeam->SetSigmaEnergy(GetSigmaEnergy(energy));
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
  if (mConvergentSource) {
    mPencilBeam->SetEllipseXThetaRotationNorm("positive");   // convergent beam
    mPencilBeam->SetEllipseYPhiRotationNorm("positive"); // convergent beam
  } else {
    mPencilBeam->SetEllipseXThetaRotationNorm("negative");   // divergent beam
    mPencilBeam->SetEllipseYPhiRotationNorm("negative"); // divergent beam
  }
  mPencilBeam->SetTestFlag(mTestFlag);

  if (mTestFlag) {
    GateMessage("Beam", 1, "Configuration of spot No. " << mCurrentSpot << " (out of " << mTotalNumberOfSpots << ")" << Gateendl);
    GateMessage("Beam", 1, "Energy\t" << energy << Gateendl);
    GateMessage("Beam", 1, "Spot weight (\"expected number of protons\")\t" << mSpotWeight[mCurrentSpot] << Gateendl);
    GateMessage("Beam", 1, "Number of protons to generate\t" << mNbProtonsToGenerate[mCurrentSpot] << Gateendl);
    GateMessage("Beam", 1, "Total Spot weight\t" << mTotalNbProtons << Gateendl);
    GateMessage("Beam", 1, "SetEnergy\t" << GetEnergy(energy) << Gateendl);
    GateMessage("Beam", 1, "SetSigmaEnergy\t" << GetSigmaEnergy(energy) << Gateendl);
    GateMessage("Beam", 1, "SetSigmaX\t" << GetSigmaX(energy) << Gateendl);
    GateMessage("Beam", 1, "SetSigmaY\t" << GetSigmaY(energy) << Gateendl);
    GateMessage("Beam", 1, "SetSigmaTheta\t" << GetSigmaTheta(energy) << Gateendl);
    GateMessage("Beam", 1, "SetSigmaPhi\t" << GetSigmaPhi(energy) << Gateendl);
    GateMessage("Beam", 1, "SetEllipseXThetaArea\t" << GetEllipseXThetaArea(energy) << Gateendl);
    GateMessage("Beam", 1, "SetEllipseYPhiArea\t" << GetEllipseYPhiArea(energy) << Gateendl);
  }
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::ConvertMuToProtons(double weight, double energy) {
  //this function introduces a dependence on energy for the spot intensities.
  //depending on whether mSpotIntensityAsNbProtons is set, the MSW[MU] or the #protons set by this function are used as a PDF.
  //mDistriGeneral takes a PDF to compute the number of protons that Gate is actually going to simulate.
  double K=37.60933;
  double SP=9.6139E-09*pow(energy,4)-7.0508E-06*pow(energy,3)+2.0028E-03*pow(energy,2)-2.7615E-01*pow(energy,1)+2.0082E+01*pow(energy,0);
  double PTP=1;
  double Gain=3./(K*SP*PTP*1.602176E-10);
  return (weight*Gain);
}
//------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::LoadClinicalBeamProperties() {

  std::string oneline;
  int PolOrder;

  std::ifstream inFile(mSourceDescriptionFile);
  if (! inFile) {
    GateError("Cannot open source description file!");
  }

  for (int i=0; i<4; i++) std::getline(inFile,oneline);
  // distance source patient
  mDistanceSourcePatient=atof(oneline.c_str());

  for (int i=0; i<2; i++) std::getline(inFile,oneline);
  // distance SMX patient
  mDistanceSMXToIsocenter=atof(oneline.c_str());

  for (int i=0; i<2; i++) std::getline(inFile,oneline);
  // distance SMY patient
  mDistanceSMYToIsocenter=atof(oneline.c_str());

  for (int i=0; i<5; i++) std::getline(inFile,oneline);
  // Energy
  PolOrder=atoi(oneline.c_str());
  mEnergy.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
      std::getline(inFile,oneline);
    mEnergy.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<4; i++) std::getline(inFile,oneline);
  // Energy
  PolOrder=atoi(oneline.c_str());
  mEnergySpread.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mEnergySpread.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<5; i++) std::getline(inFile,oneline);
  // X
  PolOrder=atoi(oneline.c_str());
  mX.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mX.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<3; i++) std::getline(inFile,oneline);
  // Theta
  PolOrder=atoi(oneline.c_str());
  mTheta.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mTheta.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<3; i++) std::getline(inFile,oneline);
  // Y
  PolOrder=atoi(oneline.c_str());
  mY.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mY.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<3; i++) std::getline(inFile,oneline);
  // Phi
  PolOrder=atoi(oneline.c_str());
  mPhi.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mPhi.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<5; i++) std::getline(inFile,oneline);
  // Emittance X Theta
  PolOrder=atoi(oneline.c_str());
  mXThetaEmittance.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mXThetaEmittance.push_back(atof(oneline.c_str()));
  }

  for (int i=0; i<3; i++) std::getline(inFile,oneline);
  // Emittance Y Phi
  PolOrder=atoi(oneline.c_str());
  mYPhiEmittance.push_back(PolOrder);
  std::getline(inFile,oneline);
  for (int i=0; i<=PolOrder; i++) {
    std::getline(inFile,oneline);
    mYPhiEmittance.push_back(atof(oneline.c_str()));
  }

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
  }
}
//------------------------------------------------------------------------------------------------------


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
G4int GateSourceTPSPencilBeam::GeneratePrimaries( G4Event* event ) {
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << Gateendl);
  G4int numVertices = 0;
  GenerateVertex( event );
  numVertices++;
  return numVertices;
}
/*
//------------------------------------------------------------------------------------------------------
G4ThreeVector GateSourceTPSPencilBeam::SetGantryRotation(G4ThreeVector v, double theta){
G4double a,b;
a=v[1]*cos(theta)-v[2]*sin(theta);
b=v[1]*sin(theta)+v[2]*cos(theta);
v[1]=a; v[2]=b;
return v;
}
*/
// FUNCTION
//------------------------------------------------------------------------------------------------------
void ReadLineTo3Doubles(double *toto, const std::string &data) {
  std::istringstream iss(data);
  std::string token;
  for (int j=0; j<3; j++) {
    getline(iss, token, ' ');
    toto[j]=atof(token.c_str());
    //  G4cout<<"toto "<<toto[j]<< Gateendl;
  }
}
// vim: ai sw=2 ts=2 et
