#ifndef GateRTPhantom_H
#define GateRTPhantom_H

//#include "GateRTPhantomMessenger.hh"
//#include "GateVGeometryVoxelReader.hh"
//#include "GateVSourceVoxelReader.hh"
//#include "GateVObjectInserter.hh"
//#include "GateSourceVoxellized.hh"

//#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateRTPhantomMessenger;
class GateVGeometryVoxelReader;
class GateVSourceVoxelReader;
class GateVVolume;

class GateRTPhantom {
public:

  GateRTPhantom(G4String name = "" ); 

  virtual ~GateRTPhantom();
  void SetVerboseLevel(G4int val) { m_VerboseLevel = val; };
  G4int GetVerboseLevel() { return m_VerboseLevel; };


  void SetTimeActivTables( G4String aFName);

  //! Provides a description of the properties of the Mgr and of its output modules 
  void Describe();

  G4int IsAttachedTo(GateVVolume*);

  //! Getter used by the Messenger to construct the commands directory
  inline G4String GetName()              { return m_name; };

  inline void     SetName(G4String name) { m_name = name; };

GateVVolume* GetInserter() { return m_inserter; };

  void SetGReader(GateVGeometryVoxelReader* aReader){ itsGReader = aReader;};

  GateVGeometryVoxelReader* GetGReader(){return itsGReader;};

  void SetSReader(GateVSourceVoxelReader* aReader){ itsSReader = aReader;};

  GateVSourceVoxelReader* GetSReader(){return itsSReader;};

  virtual void Compute(G4double aTime);

// methods for voxellized RTPhantoms

  G4int GetVoxelNx(){ return XDIM; };
  G4int GetVoxelNy(){ return YDIM; };
  G4int GetVoxelNz(){ return  ZDIM_OUTPUT; };

  G4ThreeVector GetPixelWidth(){ return pixel_width; };

//G4float  GetPixelWidth(){ return pixel_width; };


  void AttachToGeometry(G4String aname);
  void AttachToSource(G4String aname);
  void Enable() { IsEnabled = 1;};
  void SetVoxellized() { IsVoxellized = 1;};
  G4int GetVoxellized() { return IsVoxellized;};
  G4int GetEnabled() { return IsEnabled;};
  G4int IsInit() { return IsInitialized;};
  void Init(){ IsInitialized = 1; };

   void Disable();

 protected:


  //! Verbose level

  G4bool isFirst; // first computation ?

  G4int XDIM;
  G4int YDIM,ZDIM;
  G4int ZDIM_OUTPUT;

  G4ThreeVector pixel_width;

//G4double pixel_width;


  G4int                      m_VerboseLevel;

  GateRTPhantomMessenger*    m_messenger;

  G4int p_cK;   // get previous cK value  start value is 1 set in .cc;

G4int IsVoxellized; 

G4int IsEnabled;

G4int IsInitialized;

GateVVolume *m_inserter; // inserter to which the RTPhantom is attached to

GateVGeometryVoxelReader* itsGReader;
 
/* the Geometry Voxel Reader attached to the RTPhantom to Fill in Voxel material*/

GateVSourceVoxelReader* itsSReader; 

/* the Source Voxel Reader attached to the RTPhantom to Fill in Voxel activity*/

  //! class name, used by the messenger

  G4String                   m_name;

};

#endif
