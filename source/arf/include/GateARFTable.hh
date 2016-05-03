/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See GATE/LICENSE.txt for further details
 ----------------------*/

#include "GateConfiguration.h"
#include "GateMessageManager.hh"
#ifdef G4ANALYSIS_USE_ROOT
#ifndef GateARFTable_h
#define GateARFTable_h
#include"globals.hh"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"
class TH2I;
class TH2D;
class TH1D;
class GateARFTable
  {
private:
  G4String mArfTableName; /* the name of the ARF Table */
  G4int mArfTableIndex; /* the index of the AERF Table because many are defined for one simulation */
  G4double* mArfTableVector; /* contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index */
  G4double* mDrfTableVector; /* contains the ARF table before normalization */
  G4int m_drfdimx;
  G4double mfDistSrc2Img;
  G4int mAvgPixNum;
  G4double mDrfBinSize;
  G4double mLowX;
  G4double mLowY;
  G4int mDrfDimY;
  G4int mNbOfCosTheta; /* the number of discretized values of cos(theta) */
  G4int mNbOfTanPhi; /* the number of discretized values of tan(phi) */
  G4int mTotalNbOfThetaPhi; /* the product of m_NbOfcosTheta by m_NbOftanPhi */
  G4double mELow; /* the left end energy of the energy window */
  G4double mEHigh; /* the right end energy of the energy window */
  G4int mIsPrimary;
  G4double mEnergyLowOut; /* window energy specified by the user */
  G4double mEnergyHighOut; /* window energy specified by the user */

  G4double* mTheta;
  G4double* mCosTheta;
  G4double* mCosThetaI;
  G4double* mTanPhi;
  G4double* mTanPhiI;
  G4double* mPhi;
  G4double mStep1;
  G4double mStep2;
  G4double mStep3;
  G4double mStep4;
  G4double mTanPhiStep; /* setpes parameters for the theta,phi grids */
  G4double mBase1;
  G4double mBase2;
  G4double mBase3;
  G4double mBase4;
  G4double mEnergyResolution;
  G4double mEnergyReference;
  G4double mConstantTerm;
  G4double mLinearTerm;
  G4double mTotalNumberOfPhotons; /* total number of simulated photons for this incident energy window */
  long unsigned int mBinnedPhotonCounter; /*  the number of binned photons */
  int mPhiCounts;

public:
  GateARFTable(const G4String & aName);
  ~GateARFTable();
  G4String GetName()
    {
    return mArfTableName;
    }
  ;
  void GetARFAsBinaryBuffer(G4double*&);
  void FillTableFromBuffer(G4double*&);

  G4int GetPrimary()
    {
    return mIsPrimary;
    }
  ;
  void SetPrimary()
    {
    mIsPrimary = 1;
    }
  void SetNoPrimary()
    {
    mIsPrimary = 0;
    }
  G4double GetEWlow()
    {
    return mEnergyLowOut;
    }
  ;
  G4double GetEWhigh()
    {
    return mEnergyHighOut;
    }
  ;
  void SetName(const G4String & aName)
    {
    mArfTableName = aName;
    }
  ;
  G4int GetIndex()
    {
    return mArfTableIndex;
    }
  ;
  void SetIndex(const G4int & aIndex)
    {
    mArfTableIndex = aIndex;
    }
  ;
  G4int GetNbofTheta()
    {
    return mNbOfCosTheta;
    }
  ;
  G4int GetNbofPhi()
    {
    return mNbOfTanPhi;
    }
  ;
  G4int GetTotalNb()
    {
    return mTotalNbOfThetaPhi;
    }
  ;
  G4double GetElow()
    {
    return mELow;
    }
  ;
  void SetElow(const G4double & aD)
    {
    mELow = aD;
    }
  ;
  G4double GetEhigh()
    {
    return mEHigh;
    }
  ;
  void SetEhigh(const G4double & aD)
    {
    mEHigh = aD;
    }
  ;
  void Describe();
  void Initialize(const G4double & energyLow, const G4double & energyHigh);
  G4int GetIndexes(const G4double & x, const G4double & y, G4int& theta, G4int& phi);
  void NormalizeTable();
  G4double RetrieveProbability(const G4double & x, const G4double & y);
  void SetEnergyReso(const G4double & aE);
  void SetERef(const G4double & aE);
  inline G4double GetEnergyReso()
    {
    return mEnergyResolution;
    }
  ;
  inline G4double GetERef()
    {
    return mEnergyReference;
    }
  ;
  void FillDRFTable(const G4double & meanE, const G4double & X, const G4double & Y);
  void convertDRF2ARF();
  G4double computeARFfromDRF(const G4double & xI, const G4double & yJ, const G4double & cosTheta);
  void SetDistanceFromSourceToDetector(const G4double & aD)
    {
    mfDistSrc2Img = aD;
    }
  ;
  void SetNSimuPhotons(const G4double & N)
    {
    mTotalNumberOfPhotons = N;
    G4cout << " TOTAL number of photons   "
           << (long unsigned int) (mTotalNumberOfPhotons)
           << Gateendl;
           };
         };
#endif

#endif /* G4ANALYSIS_USE_ROOT */
