

#ifndef GateCSinglesFileReader_h
#define GateCCSinglesFileReader_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "globals.hh"
#include <queue>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include "GateCCRootDefs.hh"
#include "GatePulse.hh"





class GateCCSinglesFileReader 
{
public:

  static GateCCSinglesFileReader* GetInstance(G4String filen);

  ~GateCCSinglesFileReader();       //!< Public destructor

private:
  GateCCSinglesFileReader(G4String file);        //!< Private constructor: this function should only be called from GetInstance()

public:
  //! It opens the ROOT input file, sets up the singles  tree, and loads the first single
  void PrepareAcquisition();

 
  G4bool HasNextEvent();

 G4int PrepareNextEvent();
  //! This method is meant to be called by output manager before calling the methods RecordEndOfEvent() of the output modules.
  //! It creates a new hit-collection, based on the queue of hits previously filled by PrepareNextEvent()
  GatePulseList* PrepareEndOfEvent();


  //! This method must be called (normally by the application manager) after completion of a DigiGate acquisition
  //! It closes the ROOT input file
  void TerminateAfterAcquisition();

  //! Get t file name
  const  G4String& GetFileName()             { return m_fileName; };


  /*! \brief Overload of the base-class virtual method to print-out a description of the reader

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void Describe(size_t indent=0);

protected:

  //! Reads a set of Single data from the Tree, and stores them into the root-Single buffer
  void LoadSinglesData();

protected:

  G4String    	      m_fileName;     	      //!< Name of the input hit-file
  G4String             m_filePath;


  TFile*              m_singlesFile;       	      //!< the input hit file

  TTree*              m_singlesTree;       	      //!< the input hit tree
  Stat_t       	      m_entries;      	      //!< Number of entries in the tree
  G4int       	      m_currentEntry; 	      //!< Current entry in the tree


  GateCCRootSingleBuffer       m_singleBuffer;       	      //!< Buffer to store the data read from the single-tree
  //!< Each field of this structure is a buffer for one of the branches of the tree
  //!< The single-data are loaded into this buffer by LoadSingleData()
  //!< They are then transformed into a crystal-hit by PrepareNextEvent()

  std::queue<GatePulse*> m_singleQueue;   //!< Queue of waiting singles for the current event
  //!< For each event, the queue is filled  at
  //!< the beginning of each event by PrepareNextEvent(). It is emptied into
  //!< a GatePulseList at the end of each event by PrepareEndOfEvent()


  GatePulseList* pList; //
  

private:
  static GateCCSinglesFileReader*   instance;       //!< Instance of the GateHitFielReader singleton
};
//-----------------------------------------------------------------------------

#endif
#endif
