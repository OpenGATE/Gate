/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDigitizerMgr

	GateDigitizerOld highly inspired
	BaseName -> DigitizerName/CollectionName
	PulseProcessorChain -> Digitizer

	StoreNewPulseProcessorChain -> AddNewSinglesDigitizer

	DigitizerName = CollectionName

	DigiMakerModule -> Digitizer
	Digitizer is a List/Chain of a DigiModules in order to get a specific uses SinglesDigiCollection
*/

#ifndef GateDigitizerMgr_h
#define GateDigitizerMgr_h 1

#include "globals.hh"
#include "GateClockDependent.hh"
#include "GateTools.hh"
#include "GateCrystalSD.hh"
#include "G4DigiManager.hh"
#include "GateCoincidenceSorter.hh"

#include "GateDigitizerInitializationModule.hh"
#include "GateSinglesDigitizer.hh"

#include "GateCoinDigitizerInitializationModule.hh"
#include "GateCoincidenceDigitizer.hh"


class GateDigitizerMgrMessenger;
class GateVSystem;




class GateDigitizerMgr : public GateClockDependent, public G4DigiManager
{
public:
	static GateDigitizerMgr* GetInstance();

protected:
	GateDigitizerMgr();

public:
  ~GateDigitizerMgr();
  void Initialize();

  void GetDigiCollection();

  //mhadi_add[
    // Next methods were added for the multi-system approach
  inline GateSystemList* GetSystemList() const { return m_systemList; }

   // Add a system to the DigitizerMgr systems list
   virtual void AddSystem(GateVSystem* aSystem);
   // To find a system from the DigitizerMgr systems list
   GateVSystem* FindSystem(GateSinglesDigitizer* digitizer);
   GateVSystem* FindSystem(G4String& systemName);

   //Messenger commands
   inline const G4String&  GetElementTypeName()
   { return m_elementTypeName;}

    GateClockDependent* FindElement(G4String baseName);
   //	   { return FindElement( MakeElementName(baseName) ) ; }
   //

    void AddNewSD(GateCrystalSD*);

   void ShowSummary();
   /// Methods for Singles
   //! Run Singles Digitizers
   void RunDigitizers();
   //! Integrates a new digitizer/singlesCollection
   void AddNewSinglesDigitizer(GateSinglesDigitizer* digitizer);
   //! Find Digitizer by its name
  GateSinglesDigitizer* FindSinglesDigitizer(G4String mName);
   /// End of methods for Singles




   /// Methods for Coincidences

   //Sorters
   //Sorters -> Initialization of Coincidence Digitizers

   void RunCoincidenceSorters();
   void AddNewCoincidenceSorter(GateCoincidenceSorter* coincidenceSorter);
   GateCoincidenceSorter* FindCoincidenceSorter(G4String mName);
   //Coincidence digitizers
   void RunCoincidenceDigitizers();
   void AddNewCoincidenceDigitizer(GateCoincidenceDigitizer* coincidenceDigitizer);
   GateCoincidenceDigitizer* FindCoincidenceDigitizer(G4String mName);
   /// End of methods for Coincidences

private:


  GateDigitizerMgrMessenger *fMessenger;

  static GateDigitizerMgr*  theDigitizerMgr;

protected:
  G4String 					m_elementTypeName;	 //!< Type-name for DigitizerMgr --> used only for cout and help messengers
  GateSystemList*                               m_systemList;            //! List of systems to which the DigitizerOld is attached
  G4int m_collectionID;

public:
  G4bool m_isInitialized;
  G4bool m_isTheFirstEvent;
  std::vector<GateCrystalSD*>    	m_SDlist;	 //!< Vector of Sensitive Detectors
  std::vector<GateDigitizerInitializationModule*>    	m_digitizerIMList;	 //!< Vector of initialisation modules for different SD
  std::vector<GateSinglesDigitizer*>    	m_SingleDigitizersList;	 //!< Vector of digitizers
  std::vector<GateCoincidenceSorter*>    	m_CoincidenceSortersList;	 //!< Vector of coincidence sorters


  std::vector<GateCoincidenceDigitizer*>    	m_CoincidenceDigitizersList;	 //!< Vector of coincidence digitizers
  std::vector<GateCoinDigitizerInitializationModule*> m_coinDigitizerIMList;

  G4bool m_recordSingles;
  G4bool m_recordCoincidences;

  G4bool m_alreadyRun;


};
#endif
