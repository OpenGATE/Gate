

#ifndef GateSequenceCoincidenceTreeReader_h
#define GateSequenceCoincidenceTreeReader_h 1

#include "GateComptonCameraCones.hh"

//root libraries
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TObject.h"

#include "GateCCRootDefs.hh"



class GateSequenceCoincidenceTreeReader
{
public:
    GateSequenceCoincidenceTreeReader(const std::string path);
     void PrepareAcquisition();
      void LoadCoincData();
     bool hasNext();
 
    G4int PrepareNextEvent();
    G4int PrepareNextEventIdeal();
    GateComptonCameraCones PrepareEndOfEvent();


private:

    const std::string filepath;


      TFile * f;
      TTree * m_coincTree;
      int nentries;
      int currentEntry;
      int oldCoincID;


      GateCCRootCoincBuffer       m_coincBuffer;


      GateComptonCameraCones aCone;

    



};

#endif 








