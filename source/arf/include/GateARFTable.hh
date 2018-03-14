/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
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
  G4String _ArfTableName; /* the name of the ARF Table */
  G4int _ArfTableIndex; /* the index of the ARF Table because many are defined for one simulation */
  G4double* _ArfTableVector; /* contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index */
  G4double* _DrfTableVector; /* contains the ARF table before normalization */
  G4int _DrfTableDimensionX;
  G4int _DrfTableDimensionY;
  G4double _DistanceSourceToImage;
  G4int _AverageNumberOfPixels;
  G4double _DrfBinSize;
  G4double _LowX;
  G4double _LowY;

  G4int _NumberOfCosTheta; /* the number of discretized values of cos(theta) */
  G4int _NumberOfTanPhi; /* the number of discretized values of tan(phi) */
  G4int _TotalNumberbOfThetaPhi; /* the product of m_NbOfcosTheta by m_NbOftanPhi */
  G4double _EnergyLow; /* the left end energy of the energy window */
  G4double _EnergyHigh; /* the right end energy of the energy window */
  G4int _IsPrimary;
  G4double _EnergyLowUser; /* window energy specified by the user */
  G4double _EnergyHighUser; /* window energy specified by the user */

  G4double* _ThetaVector;
  G4double* _CosThetaVector;
  G4double* _CosThetaIVector;
  G4double* _TanPhiVector;
  G4double* mPhi;
  G4double mStep1;
  G4double mStep2;
  G4double mStep3;
  G4double mStep4;
  G4double mTanPhiStep; /* Steps parameters for the theta,phi grids */
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
    return _ArfTableName;
    }
  ;
  void GetARFAsBinaryBuffer(G4double*&);
  void FillTableFromBuffer(G4double*&);

  G4int GetPrimary()
    {
    return _IsPrimary;
    }
  ;
  void SetPrimary()
    {
    _IsPrimary = 1;
    }
  void SetNoPrimary()
    {
    _IsPrimary = 0;
    }
  G4double GetEWlow()
    {
    return _EnergyLowUser;
    }
  ;
  G4double GetEWhigh()
    {
    return _EnergyHighUser;
    }
  ;
  void SetName(const G4String & aName)
    {
    _ArfTableName = aName;
    }
  ;
  G4int GetIndex()
    {
    return _ArfTableIndex;
    }
  ;
  void SetIndex(const G4int & aIndex)
    {
    _ArfTableIndex = aIndex;
    }
  ;
  G4int GetNbofTheta()
    {
    return _NumberOfCosTheta;
    }
  ;
  G4int GetNbofPhi()
    {
    return _NumberOfTanPhi;
    }
  ;
  G4int GetTotalNb()
    {
    return _TotalNumberbOfThetaPhi;
    }
  ;
  G4double GetElow()
    {
    return _EnergyLow;
    }
  ;
  void SetElow(const G4double & aD)
    {
    _EnergyLow = aD;
    }
  ;
  G4double GetEhigh()
    {
    return _EnergyHigh;
    }
  ;
  void SetEhigh(const G4double & aD)
    {
    _EnergyHigh = aD;
    }
  ;
  void InitializePhi();
  void InitializeCosTheta();
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
  G4int GetOneDimensionIndex(G4int  x, G4int y);
  void FillDRFTable(const G4double & meanE, const G4double & X, const G4double & Y);
  void convertDRF2ARF();

  G4double computeARFfromDRF(const G4double & xI, const G4double & yJ, const G4double & cosTheta);
  void SetDistanceFromSourceToDetector(const G4double & aD)
    {
    _DistanceSourceToImage = aD;
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
