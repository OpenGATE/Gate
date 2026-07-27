/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateOfflineFileReader_h
#define GateOfflineFileReader_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "globals.hh"
#include <queue>

#include "G4Event.hh"

class G4Event;
class GateHit;


#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include "GateRootDefs.hh"
#include "GateClockDependent.hh"

class GateOfflineFileReaderMessenger;

/*! \class  GateOfflineFileReader
    \brief  Reads hits data from a ROOT simulation-output file and recreates hit-collections for digitisation

    - GateOfflineFileReader - by Daniel.Strul@iphe.unil.ch (Oct. 2002)

    - The GateOfflineFileReader is a singleton. It is designed to be used in the DigiGate mode.
      In this mode, the GateOfflineFileReader will read hits data from a ROOT simulation-output file.
      Based on these data, it will recreate hit-collections that can be fed to the digitizer to
      reprocess the hits.
*/
class GateOfflineFileReader : public GateClockDependent
{
public:
    /*! This function allows to retrieve the current instance of the GateOfflineFileReader singleton

      	If the GateOfflineFileReader already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateOfflineFileReader constructor
    */
  static GateOfflineFileReader* GetInstance();

  ~GateOfflineFileReader();       //!< Public destructor

private:
  GateOfflineFileReader();        //!< Private constructor: this function should only be called from GetInstance()

public:
  //! This method must be called (normally by the application manager) before starting a new DigiGate acquisition
  //! It opens the ROOT input file, sets up the hit tree, and loads the first hit
  void PrepareAcquisition();

  /*! \brief This method is meant to be called by the primary generator action at the beginning of each event.
      \brief It read a series of hit data from the ROOT file, and stores them into a queue of hits

      \returns 1 -> series of hits OK for the current run, 0 -> either no hits or series of hits for the NEXT run
  */
  G4int PrepareNextEventFromHits(G4Event* event);
  G4int PrepareNextEventFromSingles(G4Event* event);

  //! This method must be called (normally by the application manager) after completion of a DigiGate acquisition
  //! It closes the ROOT input file
  void TerminateAfterAcquisition();

  //! Get the hit file name
  const  G4String& GetFileName()             { return m_fileName; };
  //! Set the hit file name
  void   SetFileName(const G4String aName)   { m_fileName = aName; };

  /*! \brief Overload of the base-class virtual method to print-out a description of the reader

      \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void Describe(size_t indent=0);


  const std::vector<GateHit*>& GetHitVector() const   { return m_hitVector;}
  std::vector<GateHit*>& GetHitVector()  { return m_hitVector; }

  const std::vector<GateDigi*>& GetDigiVector() const   { return m_digiVector;}
  std::vector<GateDigi*>& GetDigiVector()  { return m_digiVector; }

    bool IsFinished() const  { return m_finished; }

protected:

  //! Reads a set of hit data from the hit-tree, and stores them into the root-hit buffer
  void LoadHitData();
  void LoadSinglesData();
protected:
  bool m_finished;
  G4String    	      m_fileName;     	      //!< Name of the input hit-file
  TFile*              m_inptuFile;       	      //!< the input hit file

  TTree*              m_inputTree;       	      //!< the input hit tree
  Stat_t       	      m_entries;      	      //!< Number of entries in the tree
  G4int       	      m_currentEntry; 	      //!< Current entry in the tree



  GateRootHitBuffer        m_hitBuffer;       	      //!< Buffer to store the data read from the hit-tree
      	      	      	      	      	      //!< Each field of this structure is a buffer for one of the branches of the tree
					      //!< The hit-data are loaded into this buffer by LoadHitData()
					      //!< They are then transformed into a crystal-hit by PrepareNextEvent()

  GateRootSingleBuffer        m_singleBuffer;       	      //!< Buffer to store the data read from the hit-tree
      	      	      	      	      	      //!< Each field of this structure is a buffer for one of the branches of the tree
					      //!< The hit-data are loaded into this buffer by LoadHitData()
					      //!< They are then transformed into a crystal-hit by PrepareNextEvent()

  std::vector<GateHit*> m_hitVector;
  std::vector<GateDigi*> m_digiVector;


  GateOfflineFileReaderMessenger *m_messenger;    //!< Messenger;

private:
  static GateOfflineFileReader*   instance;       //!< Instance of the GateHitFielReader singleton
};

#endif
#endif
