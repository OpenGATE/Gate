/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateRandomEngine_h
#define GateRandomEngine_h 1

#include "GateRandomEngineMessenger.hh"
#include "CLHEP/Random/RandomEngine.h"

class GateRandomEngineMessenger;

class GateRandomEngine
{

public:
  ~GateRandomEngine();
  //! Used to create and access the GateRandomEngine
  static GateRandomEngine* GetInstance() {
    if (instance == 0)
      instance = new GateRandomEngine();
    return instance;
  };

public:
  inline CLHEP::HepRandomEngine* GetRandomEngine() {return theRandomEngine;}
  inline G4int GetVerbosity() {return theVerbosity;}
  inline void SetVerbosity(G4int aVerbosity) {theVerbosity=aVerbosity;}
  void SetRandomEngine(const G4String& aName);
  void SetEngineSeed(const G4String& value);
  void resetEngineFrom(const G4String& file); //TC
  void ShowStatus();
  void Initialize();

private:
  // Private constructor because the class is a singleton
  GateRandomEngine();
  static GateRandomEngine* instance;
  CLHEP::HepRandomEngine* theRandomEngine;
  G4int theVerbosity;
  GateRandomEngineMessenger* theMessenger;
  G4String theSeed;
  G4String theSeedFile; //TC
};

#endif
