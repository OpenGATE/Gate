

#ifndef GateCCCoincidencesFileReader_h
#define GateCCCoincidencesFileReader_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "globals.hh"
#include <queue>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include "GateCCRootDefs.hh"
#include "GatePulse.hh"
#include "GateCoincidencePulse.hh"





class GateCCCoincidencesFileReader
{
public:

    static GateCCCoincidencesFileReader* GetInstance(G4String filen);

    ~GateCCCoincidencesFileReader();       //!< Public destructor

private:
    GateCCCoincidencesFileReader(G4String file);        //!< Private constructor: this function should only be called from GetInstance()

public:
    //! It opens the ROOT input file, sets up the singles  tree, and loads the first single
    void PrepareAcquisition();


    G4bool HasNextEvent();

    G4int PrepareNextEvent();
    GateCoincidencePulse* PrepareEndOfEvent();

    void TerminateAfterAcquisition();

    //! Get t file name
    const  G4String& GetFileName()             { return m_fileName; };


    /*! \brief Overload of the base-class virtual method to print-out a description of the reader

    \param indent: the print-out indentation (cosmetic parameter)
  */
    virtual void Describe(size_t indent=0);

protected:


    void LoadCoincData();

protected:

    G4String    	    m_fileName;     	      //!< Name of the input hit-file
    G4String            m_filePath;


    TFile*              m_coincFile;       	      //!< the input hit file
    TTree*              m_coincTree;       	      //!< the input hit tree
    Stat_t       	    m_entries;      	      //!< Number of entries in the tree
    G4int       	    m_currentEntry; 	      //!< Current entry in the tree


    GateCCRootCoincBuffer       m_coincBuffer;
    GateCoincidencePulse* m_coincidencePulse;// Queue of waiting singles for the current coincID



private:
    static GateCCCoincidencesFileReader*   instance;       //!< Instance of the GateHitFielReader singleton
};
//-----------------------------------------------------------------------------

#endif
#endif
