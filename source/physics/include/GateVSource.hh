/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEVSOURCE_H
#define GATEVSOURCE_H 1

#include <vector>

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4SingleParticleSource.hh"
//#include "G4VPrimaryGenerator.hh"
#include "G4PrimaryParticle.hh"

#include "GateSimplifiedDecay.hh"
#include "GateSPSPosDistribution.hh"
#include "GateSPSEneDistribution.hh"
#include "GateSPSAngDistribution.hh"
#include "GateVSourceMessenger.hh"
#include "GateSingleParticleSourceMessenger.hh"
#include "GateVVolume.hh"
#include "GateImage.hh"

#include "G4Colour.hh"
#include "GateMaps.hh"
//-------------------------------------------------------------------------------------------------
class GateVSource : public G4SingleParticleSource
{
public:
  GateVSource( G4String name );
  virtual ~GateVSource();

  virtual  void Initialize(){}

  virtual void SetName( G4String value ) { m_name = value; }
  virtual G4String GetName()             { return m_name; }

  virtual void SetType( G4String value ) { m_type = value; }
  virtual G4String GetType()             { return m_type; }

  virtual void SetSourceID(G4int value)  { m_sourceID = value; }
  virtual G4int GetSourceID()            { return m_sourceID; }

  virtual void SetActivity(G4double value) { m_activity = value; }
  //virtual void SetActivity();
  virtual G4double GetActivity()           { return m_activity; }

  virtual void SetStartTime(G4double value) { m_startTime = value; }
  virtual G4double GetStartTime()           { return m_startTime; }

  virtual void SetTime(G4double value) { m_time = value; }
  virtual G4double GetTime()           { return m_time; }

  virtual void SetNeedInit(G4bool value) { m_needInit = value; }
  virtual G4bool GetNeedInit()           { return m_needInit; }

  virtual void SetForcedUnstableFlag( G4bool value ) { m_forcedUnstableFlag = value; }
  virtual G4bool GetForcedUnstableFlag()             { return m_forcedUnstableFlag; }

  virtual void SetAccolinearityFlag( G4bool value ) { m_accolinearityFlag = value; }
  virtual G4bool GetAccolinearityFlag()            { return m_accolinearityFlag; }

  virtual void SetAccoValue( G4double value ) { m_accoValue = value; }
  virtual G4double GetAccoValue()             { return m_accoValue; }

  virtual void SetIfSourceVoxelized( G4bool value ) { mIsSourceVoxelized = value; };      // added by I. Martinez-Rovira (immamartinez@gmail.com)
  virtual G4bool GetIfSourceVoxelized() { return mIsSourceVoxelized; };    // added by I. Martinez-Rovira (immamartinez@gmail.com)

  // getter and setter for T_1/2, without introducing one more variable
  virtual void SetForcedHalfLife( G4double value ) { m_forcedLifeTime =  value / log( 2. ); }
  virtual G4double GetForcedHalfLife()             { return m_forcedLifeTime * log( 2. ); };
  virtual void SetIonDefaultHalfLife();
  virtual void Update(double time);
  virtual G4double GetNextTime( G4double timeStart );
  //virtual G4double GetNextTimeInSuccessiveSourceMode(G4double timeStart, G4int mNbOfParticleInTheCurrentRun);
  virtual void Dump( G4int level );
  virtual void SetVerboseLevel( G4int value ) { nVerboseLevel = value; }
  virtual G4int GetVerboseLevel()             { return nVerboseLevel; }

  // Main functions
  virtual G4int GeneratePrimaries(G4Event* event);
  virtual void GeneratePrimaryVertex(G4Event* event);

  void GeneratePrimariesForBackToBackSource(G4Event* event);
  void GeneratePrimariesForFastI124Source(G4Event* event);

  virtual GateSPSPosDistribution* GetPosDist() { return m_posSPS ; }
  virtual GateSPSEneDistribution* GetEneDist() { return m_eneSPS ; }
  virtual GateSPSAngDistribution* GetAngDist() { return m_angSPS ; }

  virtual G4ThreeVector getRotX() { return mRotX ; }
  virtual G4ThreeVector getRotY() { return mRotY ; }
  virtual G4ThreeVector getRotZ() { return mRotZ ; }

  G4String GetRelativePlacementVolume();
  void SetRelativePlacementVolume(G4String volname);
  void EnableRegularActivity(bool b);
  //  void SetTimeInterval(double time);

  void SetNumberOfParticles(int n){m_NbOfParticles=n;}
  G4int GetNumberOfParticles(){return m_NbOfParticles;}
  void SetSourceWeight(double v){m_weight=v;}
  G4double GetSourceWeight(){return m_weight;}

  void SetTimeActivityFilename(G4String filename);
  //void AddTimeSlicesFromFile(G4String filename);

  // Class function related to the userFluenceImage source
  // - initialization and 2D biased random engine
  void SetUserFluenceFilename( G4String s ) { mUserFluenceFilename = s; }
  void InitializeUserFluence();
  G4ThreeVector UserFluencePosGenerateOne();
  // - copy of useful functions from G4SPSPosDistribution (unreachable G4 class)
  void SetPosRot1(G4ThreeVector);
  void SetPosRot2(G4ThreeVector);
  void SetCentreCoords(G4ThreeVector);

    // Class function related to the userFocused angDist type
  void InitializeUserFocalShape();
  G4ThreeVector UserFocalShapeGenerateOne();
  G4SPSPosDistribution *GetUserFocalShape() { return mUserFocalShape; }
  void SetUserFocalShapeFlag(G4bool b) { mUserFocalShapeInitialisation = b; }

  //void AddTimeSlices(double time, int nParticles);
  //std::vector<double> GetTimePerSlice() {return mTimePerSlice;}
  //std::vector<int> GetNumberOfParticlesPerSlice() {return mNumberOfParticlesPerSlice;}

  void SetIntensity(G4double value){m_intensity = value;}
  G4double GetIntensity(){return m_intensity ;}

  void Visualize( G4String parms);

  void TrigMat();

private:
  typedef GateMap<G4String,G4Colour> GateColorMap ;
  typedef GateColorMap::MapPair GateColorPair ;

  static GateColorPair theColorTable[];
  static GateColorMap theColorMap;

protected:
  GateVSourceMessenger*               m_sourceMessenger;
  GateSingleParticleSourceMessenger*  m_SPSMessenger ;
  GateSPSPosDistribution*             m_posSPS;
  GateSPSEneDistribution*             m_eneSPS;
  GateSPSAngDistribution*             m_angSPS;

  void ChangeParticlePositionRelativeToAttachedVolume(G4ThreeVector & position);
  void ChangeParticleMomentumRelativeToAttachedVolume(G4ParticleMomentum & momentum);

  G4String   m_name;         // source name
  G4String   m_type;         // source type
  G4int      m_sourceID;     // source progressive number
  G4double   m_activity;     // activity of the source (e.g. # becquerel)
  G4double   m_startTime;
  G4double   m_time;
  G4bool     m_needInit;
  G4int      nVerboseLevel;
  G4bool     m_forcedUnstableFlag;
  G4double   m_forcedLifeTime;
  G4bool     m_accolinearityFlag;
  G4double   m_accoValue;
  G4String   mRelativePlacementVolumeName;
  G4String   m_materialName;
  GateVVolume * mVolume;
  G4bool    mIsSourceVoxelized;  // added by I. Martinez-Rovira (immamartinez@gmail.com)

  G4double mEnergy;
  bool mEnableRegularActivity;
  G4double m_timeInterval;
  G4double m_weight;
  G4int m_NbOfParticles;
  G4double m_intensity;


  G4double mSourceTime;
  //std::vector<double> mTimePerSlice;
  //std::vector<int> mNumberOfParticlesPerSlice;

  // Class members related to the userFluenceImage source
  // - initialization and 2D biased random engine
  G4bool mIsUserFluenceActive;
  G4String mUserFluenceFilename;
  G4ThreeVector mUserFluenceVoxelSize;
  std::vector<double> mUserPosX;
  std::vector<double> mUserPosY;
  G4SPSRandomGenerator *mUserPosGenX;
  std::vector<G4SPSRandomGenerator*> mUserPosGenY;
  // - copy of useful members from G4SPSPosDistribution (unreachable G4 class)
  G4ThreeVector mCentreCoords;
  G4ThreeVector mRotX;
  G4ThreeVector mRotY;
  G4ThreeVector mRotZ;

    //userFocalShape
  G4bool mIsUserFocalShapeActive;
  G4bool mUserFocalShapeInitialisation;
  G4double mUserFocalRadius;
  G4SPSPosDistribution *mUserFocalShape;

  std::vector<double> mTimeList;
  std::vector<double> mActivityList;

  /* PY Descourt 08/09/2009 */
    G4bool fAbortNow; // detector mode
  G4ThreeVector fPosition;// for detector mode because G4Trajectory does not allow to set first trajectory point position !!!
  G4ParticleDefinition* m_pd;

};

typedef std::vector<GateVSource*> GateVSourceVector;

#endif
