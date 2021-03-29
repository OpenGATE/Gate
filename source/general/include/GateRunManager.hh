/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

//----------------------------------------------------------------------------------------
/*! \class  GateRunManager
  \brief  Extension of the G4RunManager to provide new functions
  \brief  and automated resetting of the navigator at the beginning of each run

  - GateRunManager - by Daniel.Strul@iphe.unil.ch

  - A GateRunManager is derived from G4RunManager, and provides the following functionalities:
  - PhysicsHasChanged(): to reset the physicsInitialised flag
  - InitGeometryOnly(): new method to initialise only the geometry, allowing us to build the
  GATE geometry
  - RunInitialisation(): overload of G4RunManager()::RunInitialisation() that resets the geometry
  navigator.

  \sa GateSystemComponent, GateBoxCreatorComponent, GateArrayRepeater
*/


#ifndef GateRunManager_h
#define GateRunManager_h 1

#include "G4RunManager.hh"
#include "G4VModularPhysicsList.hh"
#include "GateHounsfieldToMaterialsBuilder.hh"

class GateRunManagerMessenger;
class GateDetectorConstruction;

class GateRunManager : public G4RunManager
{
public:
  //! Constructor
  GateRunManager();

  //! Constructor
  virtual ~GateRunManager();

public:
  //! Reset the physicsInitialised flag to zero
  inline void PhysicsHasChanged()
  {	physicsInitialized = false; }

  //! Initialise the geometry, the actors and the physics list
  void InitializeAll();

  //! Initialise only the geometry, to allow the building of the GATE geometry
  void InitGeometryOnly();

  void EnableDecay(bool b) { mEnableDecay = b; }

  void InitPhysics();

  //! Overload of G4RunManager()::RunInitialisation() that resets the geometry navigator
  void RunInitialization();

  //! Return the instance of the run manager
  static GateRunManager* GetRunManager()
  {	return dynamic_cast<GateRunManager*>(G4RunManager::GetRunManager()); }

  bool GetGlobalOutputFlag() { return mGlobalOutputFlag; }
  void EnableGlobalOutput(bool b) { mGlobalOutputFlag = b; }
  void SetUserPhysicList(G4VModularPhysicsList * m) { mUserPhysicList = m; }
  void SetUserPhysicListName(G4String m) { mUserPhysicListName = m; }

private :

  GateDetectorConstruction* detConstruction;
  GateDetectorConstruction* det;
  GateRunManagerMessenger* pMessenger;
  bool mIsGateInitializationCalled;
  GateHounsfieldToMaterialsBuilder * mHounsfieldToMaterialsBuilder;
  bool mGlobalOutputFlag;
  G4VModularPhysicsList * mUserPhysicList;
  G4String mUserPhysicListName;
  bool mEnableDecay;
};
//----------------------------------------------------------------------------------------

#endif
