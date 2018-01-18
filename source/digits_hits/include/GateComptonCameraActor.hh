

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT


#ifndef GATECOMPTONCAMERACTOR_HH
#define GATECOMPTONCAMERACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"
#include "GateDiscreteSpectrum.hh"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

class G4EmCalculator;
//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateComptonCameraActor : public GateVActor
{
public:

  virtual ~GateComptonCameraActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateComptonCameraActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event *) ;
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void EndOfEventAction(const G4Event*);

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}



    int GetNDaughtersBB() {return nDaughterBB;}



  double GetEmin() {return mEmin; }
  double GetEmax() {return mEmax;}
  int GetENBins() {return mENBins;}
  //
  void SetEmin(double v) {mEmin = v;}
  void SetEmax(double v) {mEmax = v;}
  void SetENBins(int v) {mENBins = v;}



  double GetVolIDmin() {return mVolmin; }
  double GetVolIDmax() {return mVolmax;}
  int GetVolNBins() {return mVolNBins;}
  //

  void SetVolIDmin(double v) {mVolmin = v;}
  void SetVolIDmax(double v) {mVolmax= v;}
  void SetVolIDNBins(int v) {
      //In case world an BB are not tacking into account. If they are the only thing that happens is two mwpty bins
      mVolNBins= v+2;
      mVolmax=(double) mVolNBins;
  }



  double GetEdepmin() {return mEdepmin; }
  double GetEdepmax() {return mEdepmax;}
  int GetEdepNBins() {return mEdepNBins;}
//
  void SetEdepmin(double v) {mEdepmin = v;}
  void SetEdepmax(double v) {mEdepmax = v;}
  void SetEdepNBins(int v) {mEdepNBins = v;}



  void SetSaveAsTextFlag(bool b) { mSaveAsTextFlag = b; }

  //Hay que pasarlas a privadas Temporal
  double* edepInEachLayerEvt;

   double* xPos_InEachLayerEvt;
    double* yPos_InEachLayerEvt;
     double* zPos_InEachLayerEvt;
  std::vector<G4String> layerNames;
   std::vector<std::unique_ptr<TTree>> pSingles2;

protected:
  GateComptonCameraActor(G4String name, G4int depth=0);

  TFile * pTfile;

  G4String mHistName;

  TH1D * pEnergySpectrum;
  TH1D * pEdep;
  TH1D * pEdepTrack;

    TH1D * pEdepAbs;

   TH1D * pVolumeName;
  

  double mVolmin;
  double mVolmax;
  int mVolNBins;

  
  double mEmin;
  double mEmax;
  int mENBins;

  double mEdepmin;
  double mEdepmax;
  int mEdepNBins;

unsigned int nDaughterBB;
//No lo consigome da cada vez uno segun donde este
G4String attachPhysVolumeName;

  double Ei,Ef;
  int nTrack;
  int nEvent;
  bool newEvt;
  bool newTrack;
  double sumNi;


  G4ThreeVector hitPostPos;
  G4ThreeVector hitPrePos;
   double edepStep;

  double edep;
  double tof;
  double edepTrack;
  //int copyNoVolStep;
  G4String VolNameStep;




 double edptempAb;

  GateActorMessenger* pMessenger;
 int counterConstructF;

  GateDiscreteSpectrum mDiscreteSpectrum;
  void SaveAsText(TH1D * histo, G4String initial_filename);
  bool mSaveAsTextFlag;
  bool mSaveAsDiscreteSpectrumTextFlag;
  bool mEnableLETSpectrumFlag;
  
  G4EmCalculator * emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(ComptonCameraActor,GateComptonCameraActor)


#endif /* end #define GATECOMPTONCAMERAACTOR_HH */
#endif
