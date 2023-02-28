/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateOutputMgr_H
#define GateOutputMgr_H

#include "globals.hh"

#include "G4Timer.hh"
#include "GateConfiguration.h"
#include "GateVOutputModule.hh"
#include "GateHit.hh"
#include "GatePhantomHit.hh"
#include "GateDigi.hh"
#include "GateCoincidenceDigi.hh"

class G4Run;
class G4Step;
class G4Event;
class G4Timer;
class G4UserSteppingAction;

class GateOutputMgrMessenger;
class GateVVolume;
class GateSteppingAction;
class GateVGeometryVoxelStore;


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateOutputMgr
{
public:

  virtual ~GateOutputMgr(); //!< Destructor

  //! Called by GateApplicationMgr
  /*! It calls in turn the RecordBeginOfAcquisition method of the inserted modules */
  void RecordBeginOfAcquisition();
  //! Called by GateApplicationMgr
  /*! It calls in turn the RecordEndOfAcquisition method of the inserted modules */
  void RecordEndOfAcquisition();

  //! Called by GateRunAction
  /*! It calls in turn the RecordBeginOfRun method of the inserted modules */
  void RecordBeginOfRun(const G4Run *);
  //! Called by GateRunAction
  /*! It calls in turn the RecordEndOfRun method of the inserted modules */
  void RecordEndOfRun(const G4Run *);

  //! Called by GateEventAction
  /*! It calls in turn the RecordBeginOfEvent method of the inserted modules */
  void RecordBeginOfEvent(const G4Event *);
  //! Called by GateEventAction
  /*! It calls in turn the RecordEndOfEvent method of the inserted modules */
  void RecordEndOfEvent(const G4Event *);

  //! Called by GateSteppingAction
  /*! It calls in turn the RecordStep method of the inserted modules */
  void RecordStepWithVolume(const GateVVolume * , const G4Step *);

  //! Called by the Voxel Store, saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *);

  //! Used internally to add a module to the output module list
  /*! For the moment it's a simple push_back in a vector of pointers, but it may become more complex
   if the structure of the module list evolves */
  void AddOutputModule(GateVOutputModule* module);
  GateVOutputModule* FindOutputModule(G4String name);

  void BeginOfRunAction(const G4Run*);
  void EndOfRunAction(const G4Run*);

  void BeginOfEventAction(const G4Event*);
  void EndOfEventAction(const G4Event*);

  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  //! Used to create and access the OutputMgr

  static GateOutputMgr* GetInstance() {
    if (instance == 0){
      instance = new GateOutputMgr("output");}
    return instance;
  };


  inline void SetVerboseLevel(G4int val) { nVerboseLevel = val; };

  //! Provides a description of the properties of the Mgr and of its output modules
  virtual void Describe(size_t indent=0);

  //! Getter used by the Messenger to construct the commands directory
  inline G4String GetName()              { return mName; };
  inline void     SetName(G4String name) { mName = name; };

  //! Static getter and setter to know the current mode of the digitizer
  inline static DigiMode GetDigiMode()  	  { return m_digiMode;}
  inline static void SetDigiMode(DigiMode mode)   { m_digiMode = mode; }
  GateVOutputModule* GetModule(G4String);

  //! Call in startDAQ, this function search for all output module inserted
  //! in this manager to see for each enabled module if a fileName is given.
  //! If it is not the case the module is disabled and a warning is sent.
  void CheckFileNameForAllOutput();

  //! Return the current crystal-hit collection (if nay)
  GateHitsCollection*  	  GetHitCollection();
  std::vector<GateHitsCollection*> GetHitCollections();
  void SetCrystalHitsCollectionsID();
  //! Return the current phantom-hit collection (if nay)
  GatePhantomHitsCollection*  	  GetPhantomHitCollection();
  //! Return the current single-digi collection (if nay)
  GateDigiCollection*   	  GetSingleDigiCollection(const G4String& collectionName);
  //! Return the current coincidence-digi collection (if nay)
  GateCoincidenceDigiCollection*  GetCoincidenceDigiCollection(const G4String& collectionName);

  void RegisterNewHitsCollection(const G4String& aCollectionName,G4bool outputFlag);

  void RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag);
  void RegisterNewCoincidenceDigiCollection(const G4String& aCollectionName,G4bool outputFlag);

  inline G4bool GetSaveVoxelTuple()                  { return m_saveVoxelTuple; };
  inline void   SetSaveVoxelTuple(G4bool flag)       { m_saveVoxelTuple = flag; };

  inline void AllowNoOutput() {m_allowNoOutput=true;}

  /* PY Descourt 11/12/2008 */
  void RecordTracks(GateSteppingAction*);

//private:

protected :
  //! Private constructor (the class is a singleton)
  /*! In the constructor we build the list of the output modules to attach to the Mgr.
    In the future this will be substituted by a dynamic insertion of modules of predefined type.
    A Messenger of the type GateOutputMgrMessenger is built.
    \param name The name of the Output Manager is necessary to build the commands directory in the Messenger
    \sa AddOutputModule()
   */
  GateOutputMgr(const G4String name);
  static GateOutputMgr* instance;

  //! Verbose level
  G4int                      nVerboseLevel;

public: //OK GND 2022 moved to public to have access in GateAnalysis::RecordEndOfEvent to not run Digitizer if there is no output requires Singles
  //! List of the output modules
  std::vector<GateVOutputModule*>   m_outputModules;
protected:
  //! messenger for the Mgr specific commands
  GateOutputMgrMessenger*    m_messenger;

  //! class name, used by the messenger
  G4String                   mName;

  //! Code for the current digitizer mode (runtime or offline)
  static DigiMode    	     m_digiMode;

  //! Flag to check that an acquisition has started
  G4bool m_acquisitionStarted;

  //! Flag to say if the user allow to launch a simulation
  //! without any output nor actor
  G4bool m_allowNoOutput;

  G4bool   m_saveVoxelTuple;

  G4Timer m_timer;      	  //!< Timer
  std::vector<G4int> m_HCIDs;

};

#endif
