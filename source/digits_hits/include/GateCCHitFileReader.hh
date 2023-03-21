/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCCHitFileReader
*/

#ifndef GateCCHitFileReader_h
#define GateCCHitFileReader_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "globals.hh"
#include <queue>
#include <G4Event.hh>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include "GateCCRootDefs.hh"
#include "GateClockDependent.hh"
#include "GateHit.hh"

class G4Event;
class GateHit;
class GateCCHitFileReaderMessenger;

//-----------------------------------------------------------------------------
/*! \class  GateCCHitFileReader
  \brief  Reads hits data from a ROOT simulation-output file and recreates hit-collections for digitisation

  - The GateCCHitFileReader is a singleton. It is designed to be used in the DigiGate mode.
  In this mode, the GateCCHitFileReader will read hits data from a ROOT simulation-output file.
  Based on these data, it will recreate hit-collections that can be fed to the digitizer to
  reprocess the hits.
*/
class GateCCHitFileReader : public GateClockDependent
{
public:
    /*! This function allows to retrieve the current instance of the GateCCHitFileReader singleton

    If the GateCCHitFileReader already exists, GetInstance only returns a pointer to this singleton.
    If this singleton does not exist yet, GetInstance creates it by calling the private
    GateCCHitFileReader constructor
  */

    static GateCCHitFileReader* GetInstance(G4String filen);

    ~GateCCHitFileReader();       //!< Public destructor

private:
    GateCCHitFileReader();        //!< Private constructor: this function should only be called from GetInstance()
    GateCCHitFileReader(G4String file);        //!< Private constructor: this function should only be called from GetInstance()

public:
    //! This method must be called (normally by the application manager) before starting a new DigiGate acquisition
    //! It opens the ROOT input file, sets up the hit tree, and loads the first hit
    void PrepareAcquisition();

    /*! \brief This method is meant to be called by the primary generator action at the beginning of each event.
    \brief It read a series of hit data from the ROOT file, and stores them into a queue of hits

    \returns 1 -> series of hits OK for the current run, 0 -> either no hits or series of hits for the NEXT run
  */
    G4int PrepareNextEvent();
    G4bool HasNextEvent();

    //GateHitsCollection* PrepareEndOfEvent();
    std::vector<GateHit*> PrepareEndOfEvent();

    void TerminateAfterAcquisition();

    //! Get the hit file name
    const  G4String& GetFileName()             { return m_fileName; };
    //! Set the hit file name
    void   SetFileName(const G4String aName)   { m_fileName = aName; };

    /*! \brief Overload of the base-class virtual method to print-out a description of the reader

    \param indent: the print-out indentation (cosmetic parameter)
  */
    virtual void Describe(size_t indent=0);

protected:

    //! Reads a set of hit data from the hit-tree, and stores them into the root-hit buffer
    void LoadHitData();

protected:

    G4String    	      m_fileName;     	      //!< Name of the input hit-file
    G4String             m_filePath;
    bool IsCCDigi;
    TFile*              m_hitFile;       	      //!< the input hit file

    TTree*              m_hitTree;       	      //!< the input hit tree
    Stat_t       	      m_entries;      	      //!< Number of entries in the tree
    G4int       	      m_currentEntry; 	      //!< Current entry in the tree


    GateCCRootHitBuffer        m_hitBuffer;       	      //!< Buffer to store the data read from the hit-tree
    //!< Each field of this structure is a buffer for one of the branches of the tree
    //!< The hit-data are loaded into this buffer by LoadHitData()
    //!< They are then transformed into a crystal-hit by PrepareNextEvent()

    //std::queue<GateHit> m_hitQueue;   //!< Queue of waiting hits for the current event
    //!< For each event, the queue is filled (from data read out of the hit-file) at
    //!< the beginning of each event by PrepareNextEvent(). It is emptied into
    //!< a crystal-hit collection at the end of each event by PrepareEndOfEvent()

    GateCCHitFileReaderMessenger *m_messenger;    //!< Messenger;


    std::vector<GateHit*> vHitsCollection; //Hit collection


private:
    static GateCCHitFileReader*   instance;       //!< Instance of the GateHitFielReader singleton
};
//-----------------------------------------------------------------------------

#endif
#endif
