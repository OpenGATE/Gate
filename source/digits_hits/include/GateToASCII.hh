/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateToASCII_H
#define GateToASCII_H


#include <vector>
#include <fstream>

#include "GateVOutputModule.hh"

#ifdef G4ANALYSIS_USE_FILE

class GateToASCIIMessenger;
class GateVVolume;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToASCII :  public GateVOutputModule
{
public:
  GateToASCII(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode);

  virtual ~GateToASCII();
  const G4String& GiveNameOfFile();

  //! It opens the ASCII files
  void RecordBeginOfAcquisition();
  //! It closes the ASCII files.
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  //! It saves the run-specific info in the Run file
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  //! saves the Hits in the ASCII files, and calls RecordDigitizer
  void RecordEndOfEvent(const G4Event *);
  void RecordStepWithVolume(const GateVVolume * , const G4Step *);
  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *);

  //! Called by RecordEndOfEvent to store the Digis in the ASCII files
  void RecordDigitizer(const G4Event *);

  //! Nothing
  void Reset();

  class VOutputChannel
  {
    public:
      inline VOutputChannel(const G4String& aCollectionName,
			    G4bool outputFlag)
        : nVerboseLevel(0),
	  m_outputFlag(outputFlag),
	  m_fileBaseName(G4String("")),
  	  m_collectionName(aCollectionName),
	  m_fileCounter(0),
	  m_collectionID(-1)
	  //	  m_outputFileSizeLimit(2000000000),
	{}
      virtual inline ~VOutputChannel() {}

      void Open(const G4String& aFileBaseName);
      void Close();
      static void SetOutputFileSizeLimit(G4int limit) {m_outputFileSizeLimit = limit;};
      G4bool ExceedsSize();
      virtual void RecordDigitizer()=0;

      inline void SetOutputFlag(G4bool flag) { m_outputFlag = flag; };
      inline void SetVerboseLevel(G4int val) { nVerboseLevel = val; };


      G4int             nVerboseLevel;
      G4bool            m_outputFlag;
      G4String          m_fileBaseName;
      G4String          m_collectionName;
      G4int             m_fileCounter;
      long              m_outputFileBegin;
      G4int	        m_collectionID;
      std::ofstream   m_outputFile;

      static long       m_outputFileSizeLimit;
  };

  class SingleOutputChannel : public VOutputChannel
  {
    public:
      SingleOutputChannel(const G4String& aCollectionName,
			       G4bool outputFlag);
      virtual inline ~SingleOutputChannel() {}
      virtual void RecordDigitizer();
  };

  class CoincidenceOutputChannel : public VOutputChannel
  {
    public:
      CoincidenceOutputChannel(const G4String& aCollectionName,
			       G4bool outputFlag);
      virtual inline ~CoincidenceOutputChannel() {}
      virtual void RecordDigitizer();
  };

  //! flag to decide if it writes or not to the file
  G4bool GetOutFileRunsFlag()           { return m_outFileRunsFlag; };
  //! flag to decide if it writes or not to the file
  void   SetOutFileRunsFlag(G4bool flag) { m_outFileRunsFlag = flag; };

  //! flag to decide if it writes or not to the file
  G4bool GetOutFileHitsFlag()           { return m_outFileHitsFlag; };
  //! flag to decide if it writes or not to the file
  void   SetOutFileHitsFlag(G4bool flag) { m_outFileHitsFlag = flag; };

  //! flag to decide if it writes or not to the file
  G4bool GetOutFileVoxelFlag()               { return m_outFileVoxelFlag; };
  //! flag to decide if it writes or not to the file
  void   SetOutFileVoxelFlag(G4bool flag)    { m_outFileVoxelFlag = flag; };

  G4int GetRecordFlag()           { return m_recordFlag; };
  void  SetRecordFlag(G4int flag) { m_recordFlag = flag; };

  //! Get the output file name
  const  G4String& GetFileName()             { return m_fileName; };
  //! Set the output file name
  void   SetFileName(const G4String aName)   { m_fileName = aName; };

  void   RegisterNewCoincidenceDigiCollection(const G4String& aCollectionName,G4bool outputFlag);
  void   RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag);

  void SetVerboseLevel(G4int val)
  {
    GateVOutputModule::SetVerboseLevel(val);
    for (size_t i=0; i<m_outputChannelList.size(); ++i)
      m_outputChannelList[i]->SetVerboseLevel(val);
  };

private:

  G4bool   m_outFileRunsFlag;
  G4bool   m_outFileHitsFlag;
  G4bool   m_outFileVoxelFlag;
  G4int    m_recordFlag;


  GateToASCIIMessenger* m_asciiMessenger;

  std::ofstream m_outFileRun;
  std::ofstream m_outFileHits;

  G4String m_fileName;

  std::vector<VOutputChannel*>  m_outputChannelList;
};

#endif
#endif
