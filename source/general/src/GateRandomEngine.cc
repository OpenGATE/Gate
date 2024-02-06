/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateRandomEngineMessenger.hh"
#include "GateRandomEngine.hh"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/MixMaxRng.h"
#include "CLHEP/Random/MTwistEngine.h"
#include "CLHEP/Random/Ranlux64Engine.h"
#include <ctime>
#include <cstdlib>
#include <random>
#include "GateMessageManager.hh"

#ifdef G4ANALYSIS_USE_ROOT
#include "TRandom.h"
#endif

GateRandomEngine* GateRandomEngine::instance = 0;

///////////////////
//  Constructor  //
///////////////////

//!< Constructor
GateRandomEngine::GateRandomEngine()
{
  // Default
  //theRandomEngine = new CLHEP::MTwistEngine();
  theRandomEngine = new CLHEP::HepJamesRandom();
  theVerbosity = 0;
  theSeed="default";
  theSeedFile=" ";
  // Create the messenger
  theMessenger = new GateRandomEngineMessenger(this);


}

//////////////////
//  Destructor  //
//////////////////

//!< Destructor
GateRandomEngine::~GateRandomEngine()
{
  delete theRandomEngine;
  delete theMessenger;
}

///////////////////////
//  SetRandomEngine  //
///////////////////////

//!< void SetRandomEngine
void GateRandomEngine::SetRandomEngine(const G4String& aName) {
  //--- Here is the list of the allowed random engines to be used ---//
  if (aName=="JamesRandom") {
    delete theRandomEngine;
    theRandomEngine = new CLHEP::HepJamesRandom();
  }
  else if (aName=="Ranlux64") {
    delete theRandomEngine;
    theRandomEngine = new CLHEP::Ranlux64Engine();
  }
  else if (aName=="MersenneTwister") {
    delete theRandomEngine;
    theRandomEngine = new CLHEP::MTwistEngine();
  }
  else if (aName=="MixMaxRng") {
      delete theRandomEngine;
      theRandomEngine = new CLHEP::MixMaxRng();
  }
  else {
		G4String msg = "Unknown random engine '"+aName+"'. Computation aborted !!!\n";
    G4Exception( "GateRandomEngine::SetRandomEngine", "SetRandomEngine", FatalException, msg);
  }


}

/////////////////////
//  SetEngineSeed  //
/////////////////////

//!< void SetEngineSeed
void GateRandomEngine::SetEngineSeed(const G4String& value) {
  theSeed = value;
}

/////////////////////
//  SetEngineSeed from file //
/////////////////////
//!< void resetEngineFrom
void GateRandomEngine::resetEngineFrom(const G4String& file) { //TC
  theSeedFile = file;
}


//////////////////
//  ShowStatus  //
//////////////////

//!< void ShowStatus
void GateRandomEngine::ShowStatus() {
  theRandomEngine->showStatus();
}

//////////////////
//  Initialize  //
//////////////////

//!< void Initialize
void GateRandomEngine::Initialize() {
  bool isSeed = false;
  long seed = 0;
  // rest bits are additional bit used for engine initialization
  // default engine doesn't use it
  int rest = 0;

  if (theSeed=="default" && theSeedFile==" ") {
    isSeed=false;
  } else if (theSeed=="auto") {
#if __cplusplus >= 201103L
	  std::random_device rd; //this uses /dev/urandom by default
	  seed = rd(); // Generates a single random int
#else
	  // initialize seed by reading from kernel random generator /dev/urandom
	  // FIXME may not be portable
	  FILE *hrandom = fopen("/dev/urandom","rb");
	  if(fread(static_cast<void*>(&seed),sizeof(seed),1,hrandom) == 0 ){G4cerr<< "Problem reading data!!!\n";}
	  if(fread(static_cast<void*>(&rest),sizeof(rest),1,hrandom) == 0 ){G4cerr<< "Problem reading data!!!\n";}
	  fclose(hrandom);
#endif
	  isSeed=true;
  } else {
    seed = atol(theSeed.c_str());
    rest = 0;
    isSeed=true;
  }

  if (isSeed) {
    if(theSeedFile !=" " && theSeed !="default") G4Exception( "GateRandomEngine::Initialize", "Initialize", FatalException, "ERROR !! => Please: choose between a status file and a seed (defined by a number) or auto computation of initial seed!");

    if(theSeedFile == " ") {
      theRandomEngine->setSeed(seed,rest);
    } else {
      theRandomEngine->restoreStatus(theSeedFile);
    }
  }

  // use clhep engine to initialize other engine
  std::srand(static_cast<unsigned int>(*theRandomEngine));
  srandom(static_cast<unsigned int>(*theRandomEngine));

#ifdef G4ANALYSIS_USE_ROOT
  gRandom->SetSeed(static_cast<unsigned int>(*theRandomEngine));
#endif

/*
  std::cout << "***********************************\n";
  std::cout << "SEED " << seed << " " << (sizeof(seed)*8) << "bits REST " << rest << " " << (sizeof(rest)*8) << "bits\n";
  std::cout << "stdrand=" << std::rand() << " stdrandom=" << random() << Gateendl;
  std::cout << "clhep=" << theRandomEngine->name() << " seed=" << theRandomEngine->getSeed() << Gateendl;
#ifdef G4ANALYSIS_USE_ROOT
  std::cout << "root=" << gRandom->GetName() << " seed=" << gRandom->GetSeed() << Gateendl;
#endif
  std::cout << "***********************************\n";
*/

  // True initialization
  CLHEP::HepRandom::setTheEngine(theRandomEngine);
}
